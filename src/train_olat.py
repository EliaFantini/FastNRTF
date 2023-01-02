import torch
import time
import random
import os, pickle
import datetime
import pyredner
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_msssim import ssim
import lpips

from src.model import PRTNetwork, MultipleOptimizer, MultipleScheduler
from src.utils import plot, get_SHembedder, get_hashembedder, uv_to_dir, tone_loss


def train_olat(output_dir: str, data_dir: str, device, n_olat_val: int, img_res: list, batch: int, olat_iters: int, log_time_mem: bool, mlp_size: list):
    """
    Trains the PRT model with the generated OLAT images

    :param output_dir: string, path to the output folder
    :param data_dir: string, path to the data folder
    :param device: device to load the tensors on
    :param n_olat_val: int, number of OLAT images to validate the training with
    :param img_res: list of ints, [width,height] resolution of the renderings
    :param batch: number of pixels per image to train the model with in every iteration. If set to 0, the whole image will be used
    :param olat_iters: int, number of training iterations
    :param log_time_mem: bool, if True time and memory consumption are logged
    :param mlp_size: list of ints, size of the MLP used in the model. First int is the number of neurons for every layer,
                      the second int is the number of layers, the third int is the number of the layer to place a skip connection
    :return: datetime of the model to load it for next training
    """
    pyredner.set_print_timing(False)
    pyredner.set_use_gpu( True )
    # getting current date-time to distinguish saved files
    cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    # loading the list of randomly generated camera-envmap permutations used to generate olat pictures
    cam_env_idx = []
    with open(data_dir + '/_OLAT/cam_env_idx.txt', 'r') as filehandle:
        for line in filehandle:
            curr_place = list(map(lambda x: int(x), line[1:-2].split(", ")))
            cam_env_idx.append(curr_place)
    cam_env_idx_len = len(cam_env_idx)
    n_olat_train = cam_env_idx_len - n_olat_val

    mesh_dir=output_dir + '/mesh/mesh.obj'

    # initializing the hash encoder
    objects=pyredner.load_obj(mesh_dir, return_objects=True, device=device)
    vertices=objects[0].vertices.to(device)
    total_cam_num=torch.load('{}/_BUFFER/0.pt'.format(data_dir))['total_cam_num']
    for i in range(total_cam_num):
        xyz=torch.load('{}/_BUFFER/{}.pt'.format(data_dir,i))['pos_input'].to(device)[:,:3]
        vertices=torch.cat([vertices,xyz],0)

    bounding_box=[torch.min(vertices,0)[0].float()-1e-3,torch.max(vertices,0)[0].float()+1e-3]

    embed_fn, pts_ch = get_hashembedder(bounding_box,device=device,vertices=vertices.float())
    embed_fn=embed_fn.to(device)
    embedding_params = list(embed_fn.parameters())
    # initializing the spherical-harmonics encoder for out\in directions
    embedview, view_ch = get_SHembedder()
    input_features=pts_ch+ view_ch +view_ch
    # initializing MLP model
    renderer=PRTNetwork(W=mlp_size[0], D=mlp_size[1], skips=[mlp_size[2]], din=input_features, dout=3, activation='relu').to(device)
    grad_vars = list(renderer.parameters())

    # precomputing input directions' encoding
    envmap_height=16
    uni_in_dirs=torch.ones([envmap_height,envmap_height*2,3])
    for i in range(envmap_height):
        for j in range(envmap_height*2):
            x,y,z=uv_to_dir(i,j, envmap_height=envmap_height)
            uni_in_dirs[i,j,0]=x
            uni_in_dirs[i,j,1]=y
            uni_in_dirs[i,j,2]=z
    uni_in_dirs=uni_in_dirs.view(-1,3).float().to(device)

    # initializing optimizers
    lrate=5e-4
    optimizer = MultipleOptimizer([torch.optim.SparseAdam(embedding_params,lr=lrate, betas=(0.9, 0.99), eps= 1e-15),
                                               torch.optim.Adam(grad_vars,lr=lrate, betas=(0.9, 0.99), eps= 1e-15)])
    scheduler = MultipleScheduler(optimizer, 30000, gamma=0.33)
    print("Starting OLAT optimization:")

    # creating log monitor to track iterations
    monitor=output_dir + '/_OLAT/log'
    if not os.path.exists(monitor):
        os.makedirs(monitor)
    writer = SummaryWriter(log_dir=monitor)

    # freeing up some memory
    del vertices

    olat_envmap=1*torch.ones([1,1,3]).to(device)

    # initializing  metrics
    lpips_loss = lpips.LPIPS(net='alex')
    start = time.time()
    metrics = {}
    metrics['loss_train'] = []
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
    iters = []
    losses = []
    times = []

    # starting train loop
    for it in tqdm(range(olat_iters)):
        start_it = time.time()
        renderer.train()
        val = False
        if log_time_mem:
            metrics['init_mem'].append(round(torch.cuda.memory_allocated(device=device)*1e-9,4))

        # if validation iteration (every 250 iterations), deactivate gradient computation
        if it%250 == 0 and it > 0:
            val = True
            renderer.eval()
            rand_int = random.randint(n_olat_train, cam_env_idx_len - 1)
        else:
            rand_int = random.randint(0,cam_env_idx_len - n_olat_val - 1)
        # loading camera buffer containing normals, 3d positions and out directions
        cam_env=cam_env_idx[rand_int]
        cam_idx = cam_env[0]
        env_idx = cam_env[1]
        pos_input=torch.load('{}/_BUFFER/{}.pt'.format(data_dir,cam_idx))['pos_input'].to(device).float()
        mask=torch.load('{}/_BUFFER/{}.pt'.format(data_dir,cam_idx))['mask'].to(device)

        points=pos_input[:,:3]
        out_dirs=pos_input[:,3:6]
        # computing encodings
        out_dirs_ = embedview(out_dirs)
        points_ = embed_fn(points).to(device)

        Loss = 0
        psnr_loss = 0
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        in_dirs=uni_in_dirs[env_idx].view(-1,3)
        in_dir_=embedview(in_dirs)
        # loading ground truth
        olat_gt=torch.load('{}/_OLAT/{}.pt'.format(data_dir,rand_int)).to(device)

        olat_gt_m=olat_gt.view([-1,3])[mask,:]
        # if batch is set, get a random permutation of pixels to train the model with
        if batch>0 and not val:
            indices = torch.split(torch.randperm(len(points)), batch)[0]
            points_ = points_[indices, :]
            out_dirs_ = out_dirs_[indices, :]
            olat_gt_m = olat_gt_m[indices, :]
        # rendering the image
        if log_time_mem:
            metrics['pre_render_mem'].append(round(torch.cuda.memory_allocated(device=device)*1e-9,4))
        olat_radiance=renderer(points_,out_dirs_,in_dir_)
        olat_pixels=torch.sum(olat_radiance*olat_envmap,dim=-2)
        # computing loss (and psnr if validation is on)
        Loss+=tone_loss(olat_pixels,olat_gt_m)/float(points_.shape[0])

        if val:
            psnr_loss += 20*torch.log10(torch.tensor(1.0))-10*torch.log10(((olat_pixels - olat_gt_m)**2).mean(())).mean()

        losses.append(Loss.detach().item())

        if val:
            # updating metrics
            olat_img = torch.zeros([img_res[1] * img_res[0], 3]).to(device)
            olat_img[mask, :] = olat_pixels.detach()
            gt_img = torch.zeros([img_res[1] * img_res[0], 3]).to(device)
            gt_img[mask, :] = olat_gt_m.detach()
            ssim_loss = ssim(olat_img.view([img_res[1], img_res[0], 3]).permute(2,0,1).unsqueeze(0),gt_img.view([img_res[1], img_res[0], 3]).permute(2,0,1).unsqueeze(0), data_range=1.0, size_average=False)
            lpips_val = lpips_loss((-1 + (((olat_img.view([img_res[1], img_res[0], 3]).permute(2,0,1).unsqueeze(0)-
                                           torch.min(olat_img))*2)/(torch.max(olat_img) - torch.min(olat_img)))
                                    ).to('cpu'),(-1 + (((gt_img.view([img_res[1], img_res[0], 3]).permute(2,0,1).unsqueeze(0)-
                                           torch.min(gt_img))*2)/(torch.max(gt_img) - torch.min(gt_img)))).to('cpu'))

            iters.append(it)
            metrics['psnr_val'].append(psnr_loss.detach().item())
            metrics['ssim_val'].append(ssim_loss.detach().item())
            metrics['lpips_val'].append(lpips_val.detach().item())
            metrics['loss_val'].append(losses[-1])
            metrics['loss_train'].append(sum(losses)/float(len(losses)))
            print(f"\n ITER {str(it)}: loss train-val= " + "{:.5f}".format(metrics['loss_train'][-1])+"/"
                  +"{:.5f}".format(metrics['loss_val'][-1])
                  +" -- PSNR(dB)= "+"{:.5f}".format(metrics['psnr_val'][-1])
                  +" -- SSIM= " + "{:.5f}".format(metrics['ssim_val'][-1])
                  +" -- LPIPS= " + "{:.5f}".format(metrics['lpips_val'][-1])
                  +" -- avg iter time= "+"{:.4f}".format(sum(times)/float(len(times))))
            # resetting temporary data
            times = []
            losses = []

            # printing out metrics and output-groundtruth comparison
            olat_img = torch.cat([olat_img.view([img_res[1], img_res[0], 3]), gt_img.view([img_res[1], img_res[0], 3])], dim=1)
            pyredner.imwrite(olat_img.cpu(), '{}/_OLAT/{}_{}.png'.format(output_dir, cur_datetime,it))
            if not log_time_mem:
                plot(metrics, '{}/_OLAT/{}_metrics.png'.format(output_dir,cur_datetime), "log", iters)   # linear or log

        else:
            # computing gradients and updating weights
            Loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar('Loss/OLAT', Loss.item(), it)
        if log_time_mem:
            metrics['post_backward_mem'].append(round(torch.cuda.memory_allocated(device=device)*1e-9,4))

        # saving model's current state and metrics
        if it % 1000==0 and it>0:
            ckpt=output_dir + '/_OLAT/checkpoint'
            if not os.path.exists(ckpt):
                os.makedirs(ckpt)
            with open('{}/metrics_{}.pkl'.format(ckpt,cur_datetime), 'wb') as f:
                pickle.dump(metrics, f)

            torch.save({
                        'global_step': it,
                        'network_fn_state_dict': renderer.state_dict(),
                        'embed_fn_state_dict': embed_fn.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                    }, '{}/latest_{}.pt'.format(ckpt,cur_datetime))
            renderer.train()
        end_it = time.time()
        # saving time and memory consumption
        if log_time_mem:
            metrics['iter_time'].append(end_it-start_it)
            metrics['max_all_mem'].append(round(torch.cuda.max_memory_allocated(device=device)*1e-9,4))
            if it%50==0:
                print(f"\nIter time: {metrics['iter_time'][-1]:.4f}s, "
                      f"init mem:{metrics['init_mem'][-1]:.2f}GB, "
                      f"pre_render mem:{metrics['pre_render_mem'][-1]:.2f}GB, "
                      f"post_backward mem:{metrics['post_backward_mem'][-1]:.2f}GB, "
                      f"max mem:{metrics['max_all_mem'][-1]:.2f}GB")

        times.append(end_it-start_it)
    end = time.time()
    writer.add_scalar('Total time', end-start, 999999)
    print("\nOLAT training finished, total elapsed time is %.2f mn" %((end-start)/60))
    return cur_datetime




