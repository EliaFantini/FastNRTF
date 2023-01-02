import json
import pickle

import torch
import time
from torch.cuda.amp import GradScaler, autocast
import random
import os,datetime
import pyredner
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim
import lpips

from src.scene_dataset import SceneDataset
from src.utils import plot, get_SHembedder, get_hashembedder, uv_to_dir, tone_loss
from src.model import PRTNetwork, MultipleOptimizer, MultipleScheduler


def train_joint(output_dir: str, data_dir: str, device, img_res: list, batch_rgb: int, batch_olat: int, olat_model_name: str, joint_iterations: int, log_time_mem: bool, mlp_size: list, lr_env: float = 2e-3):
    """
    Trains the PRT model with both the generated OLAT images and the real captures

    :param output_dir: string, path to the output folder
    :param data_dir: string, path to the data folder
    :param device: device to load the tensors on
    :param img_res: list of ints, [width,height] resolution of the renderings
    :param batch_rgb: int, number of pixels per rgb (real capture) image to train the model with in every iteration.
    :param batch_olat: int, number of pixels per OLAT image to train the model with in every iteration. If set to 0, the whole image will be used
    :param olat_model_name: str, datetime of when the olat training was done to load its last checkpoint
    :param joint_iterations: int, number of training iterations
    :param log_time_mem: bool, if True time and memory consumption are logged
    :param mlp_size: list of ints, size of the MLP used in the model. First int is the number of neurons for every layer,
                      the second int is the number of layers, the third int is the number of the layer to place a skip connection
    :lr_env: float, learning rate for the environment map. Default is 2e-3
    :return: datetime of the model to load it for relighting tests
    """
    pyredner.set_print_timing(False)
    pyredner.set_use_gpu( True )

    # getting current date-time to distinguish saved files
    cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    # loading the training and the validation datasets
    train_dataloader=DataLoader(SceneDataset(transform_path=data_dir + "/transforms_train.json", img_res=[img_res[1], img_res[0]], nerfactor= True), batch_size=1, shuffle=True, num_workers=0)
    n_images_train = len(json.load(open(data_dir + "/transforms_train.json", 'r'))['frames'])
    val_dataset= SceneDataset(transform_path=data_dir + "/transforms_test.json", img_res=[img_res[1], img_res[0]], nerfactor= True)
    n_images_val = val_dataset.__len__()

    n_olat_cams = 0
    mesh_dir=output_dir + '/mesh/mesh.obj'
    # loading the list of randomly generated camera-envmap permutations used to generate olat pictures
    cam_env_idx = []
    with open(data_dir + '/_OLAT/cam_env_idx.txt', 'r') as filehandle:
        for line in filehandle:
            # Remove linebreak which is the last character of the string
            curr_place = list(map(lambda x: int(x), line[1:-2].split(", ")))
            # Add item to the list
            cam_env_idx.append(curr_place)
    cam_env_idx_len = len(cam_env_idx)

    # initializing hash encoder
    vertices=pyredner.load_obj(mesh_dir,return_objects=True, device=device)[0].vertices

    for i in range(torch.load('{}/_BUFFER/0.pt'.format(data_dir))['total_cam_num']):
        vertices=torch.cat([vertices, torch.load('{}/_BUFFER/{}.pt'.format(data_dir, i))['pos_input'][:, :3].to(device)], 0)
    bounding_box=[torch.min(vertices,0)[0].float() -1e-3,torch.max(vertices,0)[0].float() +1e-3]

    embed_fn, pts_ch = get_hashembedder(bounding_box,device=device,vertices=vertices.float())
    # freeing up some memory
    del bounding_box
    del vertices
    # initial spherical harmonics encoder
    embedview, view_ch = get_SHembedder()
    # initializing MLP model
    renderer=PRTNetwork(W=mlp_size[0], D=mlp_size[1], skips=[mlp_size[2]], din=pts_ch + view_ch + view_ch, dout=3, activation='relu')
    # loading the model and hash embedder checkpoints from OLAT training
    if olat_model_name is not None:
        ckpt = torch.load('{}/_OLAT/checkpoint/latest_{}.pt'.format(output_dir, olat_model_name))
        renderer.load_state_dict(ckpt['network_fn_state_dict'])
        embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])
        del ckpt
    # loading the rotated estimated envmap from nvdiffrecmc
    envmap_ini = pyredner.imread('{}/mesh/probe_rot.hdr'.format(output_dir), gamma=1).to(device)
    envmap=envmap_ini.detach()
    envmap.requires_grad=True

    # precomputing the encoding for incoming directions
    envmap_height=16
    uni_in_dirs=torch.ones([envmap_height,envmap_height*2,3], device=device)
    for i in range(envmap_height):
        for j in range(envmap_height*2):
            x,y,z=uv_to_dir(i,j, envmap_height=envmap_height)
            uni_in_dirs[i,j,0]=x
            uni_in_dirs[i,j,1]=y
            uni_in_dirs[i,j,2]=z
    uni_in_dirs=uni_in_dirs.view(-1,3)

    # initializing optimizers, losses and scheduler
    lrate=1e-4
    optimizer = MultipleOptimizer([torch.optim.SparseAdam(embed_fn.parameters(),lr=lrate, betas=(0.9, 0.99), eps= 1e-15),
                                       torch.optim.Adam(renderer.parameters(), lr=lrate, betas=(0.9, 0.99), eps=1e-15),
                                       torch.optim.Adam([envmap], lr=lr_env)]) # original 2e-4
    scheduler=MultipleScheduler(optimizer, 30000, gamma=0.33)
    l2_loss =lambda x, y : torch.sum((x - y) ** 2)
    print("Starting joint optimization:")

    # initializing monitor to log through iterations
    monitor=output_dir + '/_JOINT/log'
    if not os.path.exists(monitor):
        os.makedirs(monitor)
    writer = SummaryWriter(log_dir=monitor)
    del monitor

    uni_in_dirs_=embedview(uni_in_dirs)
    olat_envmap=1*torch.ones([1,1,3]).to(device)

    # moving to device
    renderer.to(device)
    embed_fn.to(device)

    # initializing metrics
    lpips_loss = lpips.LPIPS(net='alex')
    metrics = {}
    metrics['loss'] = []
    metrics['loss_rgb'] = []
    metrics['loss_olat'] = []
    metrics['loss_env'] = []
    metrics['loss_val'] = []
    metrics['psnr_val'] = []
    metrics['ssim_val'] = []
    metrics['lpips_val'] = []
    if log_time_mem:
        metrics['init_mem'] = []
        metrics['pre_render_mem'] = []
        metrics['post_backward_mem'] = []
        metrics['max_all_mem'] = []
        metrics['iter_time'] = []
    iters =[]
    rgb_losses=[]
    tot_losses=[]
    olat_losses=[]
    env_losses=[]
    times = []
    start=time.time()
    scaler = GradScaler()
    it = 0
    # starting training loop
    pbar = tqdm(total=joint_iterations)
    while it < joint_iterations:

        renderer.train()
        for I_idx, ground_truth in train_dataloader:
            if log_time_mem:
                metrics['init_mem'].append(round(torch.cuda.memory_allocated(device=device) * 1e-9, 4))
            # loading ground truth and precomputed normals, 3d positions and outgoing directions
            torch.cuda.empty_cache()
            I_idx= I_idx.item()
            ground_truth= ground_truth[0].to(device)
            start_it = time.time()

            pos_input=torch.load('{}/_BUFFER/{}.pt'.format(data_dir, I_idx))['pos_input'].to(device).float()
            mask=torch.load('{}/_BUFFER/{}.pt'.format(data_dir, I_idx))['mask'].to(device)

            ground_truth=ground_truth.view([1,-1,3])[:,mask,:]

            points = pos_input[:, :3].detach()
            out_dirs = pos_input[:, 3:6].detach()
            del pos_input

            # computing encodings
            out_dirs = embedview(out_dirs)
            points = embed_fn(points).float()

            # getting a random permutation of pixels to train the model with
            rnd_pixels = torch.split(torch.randperm(len(points)),batch_rgb)[0]

            optimizer.zero_grad(set_to_none=True)

            points = points[rnd_pixels, :]
            out_dirs = out_dirs[rnd_pixels, :]
            ground_truth = ground_truth[:, rnd_pixels, :].detach()

            # using pytorch's mixed precision to render the image and compute the rgb_loss
            if log_time_mem:
                metrics['pre_render_mem'].append(round(torch.cuda.memory_allocated(device=device) * 1e-9, 4))
            with autocast(dtype=torch.float16):
                rendered_pixels = torch.sum(renderer(points, out_dirs, uni_in_dirs_).unsqueeze(0) / 100 * envmap.view(1, 1, -1, 3), dim=-2)
                torch.cuda.empty_cache()
                rgb_loss = (tone_loss(rendered_pixels, ground_truth.to(device))) / float(points.shape[0])
            # picking a random pair of [cam, envmap] among those used to render OLAT training images
            rand_int = random.randint(0, cam_env_idx_len - 1)
            cam_env = cam_env_idx[rand_int]
            cam_idx = cam_env[0]
            env_idx = cam_env[1]
            del cam_env
            # loading mask, normals, 3d positions, outgoing directions
            pos_input = torch.load('{}/_BUFFER/{}.pt'.format(data_dir, cam_idx))['pos_input'].to(device).float()
            mask = torch.load('{}/_BUFFER/{}.pt'.format(data_dir, cam_idx))['mask'].to(device)
            # getting, only once, the number of randomly generated cameras used to generate OLAT images
            if n_olat_cams == 0:
                n_olat_cams = torch.load('{}/_BUFFER/{}.pt'.format(data_dir, cam_idx))['test_cam_start'] - \
                              torch.load('{}/_BUFFER/{}.pt'.format(data_dir, cam_idx))['olat_cam_start']

            points = pos_input[:, :3]
            out_dirs = pos_input[:, 3:6]
            del pos_input

            # computing encodings and loading ground truth
            out_dirs = embedview(out_dirs)
            points = embed_fn(points).float()
            in_dir = uni_in_dirs[env_idx].view(-1, 3)
            in_dir = embedview(in_dir)
            olat_gt = torch.load('{}/_OLAT/{}.pt'.format(data_dir, rand_int)).to(device)
            olat_gt = olat_gt.view([-1, 3])[mask, :]
            # if batch_olat is set, getting a random permutation of pixels to train the model with
            if batch_olat>0:
                indices = torch.split(torch.randperm(len(points)), batch_olat)[0]
                points = points[indices, :]
                out_dirs = out_dirs[indices, :]
                olat_gt = olat_gt[indices, :]
            # rendering the image and computing the olat_loss
            olat_radiance = renderer(points, out_dirs, in_dir)
            olat_pixels = torch.sum(olat_radiance * olat_envmap, dim=-2)
            olat_loss = 0.1*tone_loss(olat_pixels, olat_gt) / float(points.shape[0])
            # computing envmap's loss
            env_loss = 1e-5 * l2_loss(envmap, envmap_ini)
            if it % 20 == 0:
                dxe = envmap - torch.roll(envmap, 1, 1)
                dye = envmap - torch.roll(envmap, 1, 0)
                env_loss += 1e-6 * (torch.norm(dxe, p=1) + torch.norm(dye, p=1))
            Loss=rgb_loss+env_loss+olat_loss

            # updating metrics
            tot_losses.append(Loss.detach().item())
            env_losses.append(env_loss.detach().item())
            olat_losses.append(olat_loss.detach().item())
            rgb_losses.append(rgb_loss.detach().item())
            # computing gradients with mixed precision and updating weights
            scaler.scale(Loss).backward()

            optimizer.step(scaler)
            scheduler.step()
            scaler.update()

            if log_time_mem:
                metrics['post_backward_mem'].append(round(torch.cuda.memory_allocated(device=device) * 1e-9, 4))

            end_it = time.time()
            times.append(end_it - start_it)

            # printing and saving metrics
            writer.add_scalar('Loss/rgb', rgb_loss.item(), it)
            writer.add_scalar('Loss/OLAT', olat_loss.item(), it)
            writer.add_scalar('Loss/envmap', env_loss.item(), it)

            envmap.data=envmap.data.clamp(0,100)

            # saving time and memory consumption
            if log_time_mem:
                metrics['iter_time'].append(end_it - start_it)
                metrics['max_all_mem'].append(round(torch.cuda.max_memory_allocated(device=device) * 1e-9, 4))
                if it % 50 == 0:
                    print(f"\nIter time: {metrics['iter_time'][-1]:.4f}s, "
                          f"init mem:{metrics['init_mem'][-1]:.2f}GB, "
                          f"pre_render mem:{metrics['pre_render_mem'][-1]:.2f}GB, "
                          f"post_backward mem:{metrics['post_backward_mem'][-1]:.2f}GB, "
                          f"max mem:{metrics['max_all_mem'][-1]:.2f}GB")
            it += 1
            pbar.update(1)

        # starting validation iteration by deactivation gradient computation e freeing up some memory
        renderer.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            # choosing a random validation image and loading buffer with 3d position, mask, outgoing directions
            rand_idx = random.randint(0, n_images_val - 1)
            I_idx, ground_truth = val_dataset.__getitem__(rand_idx)
            I_idx = I_idx + n_images_train + n_olat_cams

            ground_truth= ground_truth.to(device)
            pos_input=torch.load('{}/_BUFFER/{}.pt'.format(data_dir, I_idx))['pos_input'].to(device).float()
            mask=torch.load('{}/_BUFFER/{}.pt'.format(data_dir, I_idx))['mask'].to(device)

            ground_truth=ground_truth.view([1,-1,3])[:,mask,:]

            points = pos_input[:, :3].detach()
            out_dirs = pos_input[:, 3:6].detach()
            del pos_input
            # computing encoding
            out_dirs = embedview(out_dirs)
            optimizer.zero_grad(set_to_none=True)
            points = embed_fn(points).float()

            # computing validation rgb_loss with mixed precision
            out_img = torch.zeros([img_res[1] * img_res[0], 3]).to(device)
            gt_img = torch.zeros([img_res[1] * img_res[0], 3]).to(device)
            with autocast(dtype=torch.float16):
                rendered_img = []
                # rendering the image in chunks of 1000 pixels per time to reduce memory consumption
                for pos, outd in zip(torch.split(points, 1000), torch.split(out_dirs, 1000)):
                    radiance = renderer(pos, outd, uni_in_dirs_).unsqueeze(0) / 100  # 1*N*M*3
                    rendered_pixels = torch.sum(radiance * envmap.view(1, 1, -1, 3), dim=-2).view(-1, 3)
                    rendered_img.append(rendered_pixels.detach())
                torch.cuda.empty_cache()
                # computing metrics and saving the comparison between rendering and ground truth as png
                out_img[mask, :] = torch.cat(rendered_img, dim=0)
                ground_truth = ground_truth.squeeze(0)
                rendered_img = out_img[mask, :]
                psnr_loss = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(
                    ((rendered_img - ground_truth) ** 2).mean(())).mean()

                metrics['loss_val'].append(
                    ((tone_loss(rendered_img, ground_truth)) / float(points.shape[0])).detach().item())

                gt_img[mask, :] = ground_truth.detach()

            ssim_loss = ssim(out_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0),
                             gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0), data_range=1.0,
                             size_average=False)
            lpips_val = lpips_loss((-1 + (((out_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                                            torch.min(out_img)) * 2) / (torch.max(out_img) - torch.min(out_img)))
                                    ).to('cpu'),
                                   (-1 + (((gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                                            torch.min(gt_img)) * 2) / (torch.max(gt_img) - torch.min(gt_img)))).to(
                                       'cpu'))
            out_img = torch.cat([out_img.view([img_res[1], img_res[0], 3]), gt_img.view([img_res[1], img_res[0], 3])],
                                dim=1)
            pyredner.imwrite(out_img.cpu(), '{}/_JOINT/{}_val_{}.png'.format(output_dir, cur_datetime, it))

            #updating metrics and saving them
            iters.append(it)
            metrics['psnr_val'].append(psnr_loss.detach().item())
            metrics['ssim_val'].append(ssim_loss.detach().item())
            metrics['lpips_val'].append(lpips_val.detach().item())
            len_ = float(len(tot_losses))
            metrics['loss'].append(sum(tot_losses) / len_)
            metrics['loss_rgb'].append(sum(rgb_losses) / len_)
            metrics['loss_olat'].append(sum(olat_losses) / len_)
            metrics['loss_env'].append(sum(env_losses) / len_)
            # saving envmap if it has changed
            if metrics['loss_env'][len(metrics['loss_env'])-1]!=metrics['loss_env'][len(metrics['loss_env'])-2]:
                pyredner.imwrite(envmap.cpu(), '{}/_JOINT/{}_env_{}.png'.format(output_dir, cur_datetime, it), gamma=1)
            print(f"\n ITER {str(it)}: loss train-val= " + "{:.5f}".format(metrics['loss'][-1]) + "/"
                  + "{:.5f}".format(metrics['loss_val'][-1])
                  + " -- avg rgb+olat+env losses= " + "{:.5f}".format(metrics['loss_rgb'][-1]) + "/" + "{:.5f}".format(
                metrics['loss_olat'][-1]) + "/"
                  + "{:.5f}".format(metrics['loss_env'][-1])
                  + " -- PSNR(dB)= " + "{:.5f}".format(metrics['psnr_val'][-1])
                  + " -- SSIM= " + "{:.5f}".format(metrics['ssim_val'][-1])
                  + " -- LPIPS= " + "{:.5f}".format(metrics['lpips_val'][-1])
                  + " -- avg iter time= " + "{:.4f}".format(sum(times) / float(len(times))))
            times = []
            tot_losses = []
            rgb_losses = []
            olat_losses = []
            env_losses = []
            if not log_time_mem:
                plot(metrics, '{}/_JOINT/{}_metrics.png'.format(output_dir, cur_datetime), "log", iters)  # linear or log

            # saving a checkpoint of the model
            ckpt = os.path.join(output_dir, '_JOINT/checkpoint')
            if not os.path.exists(ckpt):
                os.makedirs(ckpt)
            with open('{}/metrics_{}.pkl'.format(ckpt,cur_datetime), 'wb') as f:
                pickle.dump(metrics, f)
            torch.save({
                'global_step': it,
                'network_fn_state_dict': renderer.state_dict(),

                'Envmaps': envmap,
                'embed_fn_state_dict': embed_fn.state_dict(),

            }, '{}/{}_latest.pt'.format(ckpt, cur_datetime))
    pbar.close()
    end=time.time()
    print("\nJoint training finished, total elapsed time is %.2f mn" %((end-start)/60))

    return cur_datetime
