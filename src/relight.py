import glob
import json
import pickle
import random
import time
import lpips
import torch
import os,datetime
import pyredner
from pytorch_msssim import ssim
from tqdm import tqdm

from src.model import PRTNetwork
from src.utils import get_SHembedder, get_hashembedder, uv_to_dir, load_rgb_16bit, tone_loss


def relight(output_dir: str, data_dir: str, device, img_res: list, joint_datetime: str, num_views: int = 8, num_envmaps: int = 3, mlp_size: list = [512,7,3]):
    """
    Tests the model by relighting the object with different envmaps

    :param data_dir: string, path to the data folder
    :param output_dir: string, path to the output folder
    :param device: device to load the tensors on
    :param img_res: tuple of ints, [width,height] resolution of the renderings
    :param joint_datetime: str, datetime of when the joint training was done to load its last checkpoint
    :param num_views: int, number of cameras/views taken randomly from test/validation cameras/views to perform novel
    view synthesis. Default is 8
    :param num_envmaps: int, number of environment maps taken randomly from all Nerfactor dataset's test environment maps
             to  perform novel illumination synthesis. If set to 0, the environment map used is going to be the learned one,
             hence it's going to be just novel view synthesis, without novel illumination conditions. Max is 8. Default is 3
    :param mlp_size: list of ints, size of the MLP used in the model. First int is the number of neurons for every layer,
                    the second int is the number of layers, the third int is the number of the layer to place a skip connection
    :return: None
    """
    pyredner.set_print_timing(False)
    pyredner.set_use_gpu(True)
    # getting current date-time to distinguish saved files
    cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    # initializing settings
    cfg = json.load(open(data_dir + "/transforms_val.json", 'r')) # change to transforms_test.json for hotdog scene
    n_images = len(cfg['frames'])
    if num_views>n_images:
        num_views=n_images
    mesh_dir=output_dir + '/mesh/mesh.obj'
    n_olat_cams = torch.load('{}/_BUFFER/{}.pt'.format(data_dir, 0))['test_cam_start'] - \
                  torch.load('{}/_BUFFER/{}.pt'.format(data_dir, 0))['olat_cam_start']
    n_images_train = len(json.load(open(data_dir + "/transforms_train.json", 'r'))['frames'])

    # initializing hash encoder

    vertices=pyredner.load_obj(mesh_dir,return_objects=True, device=device)[0].vertices
    for i in range(torch.load('{}/_BUFFER/0.pt'.format(data_dir))['total_cam_num']):
        vertices=torch.cat([vertices,torch.load('{}/_BUFFER/{}.pt'.format(data_dir,i))['pos_input'][:,:3].to(device)],0)
    bounding_box=[torch.min(vertices,0)[0].float() -1e-3,torch.max(vertices,0)[0].float() +1e-3]
    embed_fn, pts_ch = get_hashembedder(bounding_box,device=device,vertices=vertices.float())
    embed_fn.to(device)
    del bounding_box
    del vertices

    # initializing spherical harmonics encoder and MLP model
    embedview, view_ch = get_SHembedder()
    renderer=PRTNetwork(W=mlp_size[0], D=mlp_size[1], skips=[mlp_size[2]], din=pts_ch + view_ch + view_ch, dout=3, activation='relu')
    renderer.to(device)

    # loading latest checkpoints from joint training to test them
    ckpt = torch.load('{}/_JOINT/checkpoint/{}_latest.pt'.format(output_dir, joint_datetime))
    renderer.load_state_dict(ckpt['network_fn_state_dict'])
    embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])


    # precomputing encodings for ingoing directions
    envmap_height=16
    uni_in_dirs=torch.ones([envmap_height,envmap_height*2,3])
    for i in range(envmap_height):
        for j in range(envmap_height*2):
            x,y,z=uv_to_dir(i,j)
            uni_in_dirs[i,j,0]=x
            uni_in_dirs[i,j,1]=y
            uni_in_dirs[i,j,2]=z

    uni_in_dirs=uni_in_dirs.view(-1,3).float().to(device)
    uni_in_dirs_=embedview(uni_in_dirs)

    # loading test envmaps
    envmaps_names=["city","courtyard","forest","interior","night","studio","sunrise","sunset"]
    envmaps = {}

    if num_envmaps>8:
        num_envmaps=8
    random.shuffle(envmaps_names)
    envmaps_names = envmaps_names[:num_envmaps]
    if num_envmaps>0:
        for name in envmaps_names:
            parent_dir = os.path.dirname(data_dir)
            path = parent_dir + '/light_probes/' + name
            files = glob.glob(path + '.hdr')
            assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
            envmap = pyredner.imread(files[0], gamma=1).to(device)
            envmap[:,:,0]=envmap[:,:,0]
            envmap[:,:,1] = envmap[:,:,1]
            envmap[:,:,2] = envmap[:,:,2]
            envmaps[name]=envmap
    else:
        envmaps['estimated'] = ckpt['Envmaps'].to(device)
    del ckpt


    # initializing metrics
    lpips_loss = lpips.LPIPS(net='alex')
    metrics = {}
    metrics['tone_loss'] = []
    metrics['psnr'] = []
    metrics['ssim'] = []
    metrics['lpips'] = []
    times = []
    # starting testing loop
    print("Starting relighting test...")
    renderer.eval()
    torch.cuda.empty_cache()

    indices = list(range(0, n_images))
    random.shuffle(indices)
    indices=indices[:num_views]
    with torch.no_grad():
        pbar = tqdm(total=len(indices)*len(envmaps.keys()))

        for i in indices:
            # loading the ground truth, mask, 3d position, outgoing directions for camera 'i'
            I_idx = i + n_images_train + n_olat_cams

            pos_input = torch.load('{}/_BUFFER/{}.pt'.format(data_dir, I_idx))['pos_input'].to(device).float()
            mask = torch.load('{}/_BUFFER/{}.pt'.format(data_dir, I_idx))['mask'].to(device)

            points=pos_input[:,:3]
            out_dirs=pos_input[:,3:6]

            # computing encodings
            out_dirs_ = embedview(out_dirs)
            points_=embed_fn(points).float()

            # for each envmap, rendering the relighted image and saving the comparison with the ground truth
            for envmap in envmaps.keys():
                out_img = torch.ones([img_res[1] * img_res[0], 3]).to(device)
                gt_img = torch.ones([img_res[1] * img_res[0], 3]).to(device)
                print(f"\nRendering view {i} with {envmap} envmap")
                Radiance = []
                start = time.time()
                for pos,outd in zip(torch.split(points_,1000),torch.split(out_dirs_,1000)):
                    radiance=renderer(pos,outd,uni_in_dirs_).unsqueeze(0)/100       #1*N*M*3
                    rendered_pixels=torch.sum(radiance*envmaps[envmap].view(1,1,-1,3),dim=-2).view(-1,3)
                    Radiance.append(rendered_pixels.detach())

                times.append(time.time()-start)
                out_img[mask,:]=torch.cat(Radiance,dim=0)
                mask= mask.to(device)
                partial_path = os.path.join(data_dir, cfg['frames'][i]['file_path'])
                if num_envmaps>0:
                    ground_truth = load_rgb_16bit(partial_path + "_" + envmap,[img_res[1], img_res[0]])
                else:
                    ground_truth = load_rgb_16bit(partial_path, [img_res[1], img_res[0]])
                gt_img[mask,:] = ground_truth.view([1,-1,3]).squeeze(dim=0).to(device)[mask,:]
                rendered_img = out_img[mask, :]
                ground_truth = gt_img[mask, :]
                # calculating metrics
                metrics['tone_loss'].append(
                    ((tone_loss(rendered_img, ground_truth)) / float(points.shape[0])).detach().item())
                psnr_loss = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(
                    ((rendered_img - ground_truth) ** 2).mean(())).mean()
                ssim_loss = ssim(out_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0),
                                 gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0), data_range=1.0,
                                 size_average=False)
                lpips_val = lpips_loss(
                    (-1 + (((out_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                             torch.min(out_img)) * 2) / (torch.max(out_img) - torch.min(out_img)))
                     ).to('cpu'),
                    (-1 + (((gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                             torch.min(gt_img)) * 2) / (torch.max(gt_img) - torch.min(gt_img)))).to(
                        'cpu'))
                metrics['psnr'].append(psnr_loss.detach().item())
                metrics['ssim'].append(ssim_loss.detach().item())
                metrics['lpips'].append(lpips_val.detach().item())
                print(f"\n Camera {str(i)}, {envmap} envmap: tone loss = " + "{:.5f}".format(metrics['tone_loss'][-1])
                      + " -- PSNR(dB)= " + "{:.5f}".format(metrics['psnr'][-1])
                      + " -- SSIM= " + "{:.5f}".format(metrics['ssim'][-1])
                      + " -- LPIPS= " + "{:.5f}".format(metrics['lpips'][-1]))

                out_img = torch.cat([out_img.view([img_res[1], img_res[0], 3]), gt_img.view([img_res[1], img_res[0], 3])], dim=1)
                pyredner.imwrite(out_img.cpu(), '{}/_RELIGHT/{}_val_{}_{}.png'.format(output_dir, cur_datetime, i,envmap))
                pbar.update(1)
    ckpt = os.path.join(output_dir, '_RELIGHT/metrics')
    # saving metrics in a pickle file and printing the averages
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)
    with open('{}/metrics_{}.pkl'.format(ckpt, cur_datetime), 'wb') as f:
        pickle.dump(metrics, f)
    pbar.close()
    print("FINISHED")
    print(f"\n AVERAGES:\n-- tone loss= " + "{:.5f}".format(sum(metrics['tone_loss'])/len(metrics['tone_loss']))
          + " -- PSNR(dB)= " + "{:.5f}".format(sum(metrics['psnr'])/len(metrics['psnr']))
          + " -- SSIM= " + "{:.5f}".format(sum(metrics['ssim'])/len(metrics['ssim']))
          + " -- LPIPS= " + "{:.5f}".format(sum(metrics['lpips'])/len(metrics['lpips']))
          + " -- Rendering time avg= " + "{:.5f}".format(sum(times)/len(times)))
    return


