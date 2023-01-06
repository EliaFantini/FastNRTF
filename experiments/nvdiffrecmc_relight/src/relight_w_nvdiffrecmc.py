import glob
import json
import pickle
import random

import lpips
import torch
import os,datetime
import pyredner
from pytorch_msssim import ssim
from tqdm import tqdm

from src.model import PRTNetwork
from src.utils import get_SHembedder, get_hashembedder, uv_to_dir, load_rgb_16bit, tone_loss


def relight_w_nvdiffrecmc(output_dir: str, data_dir: str, device, img_res: list, joint_datetime: str, num_views: int = 8, num_envmaps: int = 3, mlp_size: list = [512,7,3], scene_name: str = "Scene"):
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
    n_olat_cams = torch.load('{}/_nvBUFFER/{}.pt'.format(data_dir, 0))['test_cam_start'] - \
                  torch.load('{}/_nvBUFFER/{}.pt'.format(data_dir, 0))['olat_cam_start']
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
    envmaps_names=["city","courtyard","forest"]
    envmaps = {}

    if num_envmaps>8:
        num_envmaps=8
    if num_envmaps>0:
        for name in envmaps_names:
            path = data_dir + '/light_probes/' + name
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
    metrics['tone_loss_nv'] = []
    metrics['psnr_nv'] = []
    metrics['ssim_nv'] = []
    metrics['lpips_nv'] = []
    # starting testing loop
    print("Starting relighting test...")
    renderer.eval()
    torch.cuda.empty_cache()

    indices = list(range(0, n_images))
    with torch.no_grad():
        pbar = tqdm(total=len(indices)*len(envmaps.keys()))

        for i in indices:
            # loading the ground truth, mask, 3d position, outgoing directions for camera 'i'
            #I_idx = i + n_images_train + n_olat_cams
            I_idx = i

            pos_input = torch.load('{}/_nvBUFFER/{}.pt'.format(data_dir, I_idx))['pos_input'].to(device).float()
            mask = torch.load('{}/_nvBUFFER/{}.pt'.format(data_dir, I_idx))['mask'].to(device)

            points=pos_input[:,:3]
            out_dirs=pos_input[:,3:6]

            # computing encodings
            out_dirs_ = embedview(out_dirs)
            points_=embed_fn(points).float()

            # for each envmap, rendering the relighted image and saving the comparison with the ground truth
            for envmap in envmaps.keys():
                out_img = torch.ones([img_res[1] * img_res[0], 3]).to(device)
                nvdiffrecmc_img =  torch.ones([img_res[1] * img_res[0], 3]).to(device)
                gt_img = torch.ones([img_res[1] * img_res[0], 3]).to(device)
                print(f"\nRendering view {i} with {envmap} envmap")
                Radiance = []
                for pos,outd in zip(torch.split(points_,1000),torch.split(out_dirs_,1000)):
                    radiance=renderer(pos,outd,uni_in_dirs_).unsqueeze(0)/100       #1*N*M*3
                    rendered_pixels=torch.sum(radiance*envmaps[envmap].view(1,1,-1,3),dim=-2).view(-1,3)
                    Radiance.append(rendered_pixels.detach())

                out_img[mask,:]=torch.cat(Radiance,dim=0)
                mask= mask.to(device)
                partial_path = os.path.join(data_dir, cfg['frames'][i]['file_path'])
                if num_envmaps>0:
                    nvdiffrecmc= load_rgb_16bit(f"/scratch/2022-fall-sp-fantini/NRTF/nvdiffrecmc_relight/our_relit/" + f"{scene_name}/" + f"{envmap}_r_00{i}",[img_res[1], img_res[0]])
                    ground_truth = load_rgb_16bit(partial_path + "_" + envmap,[img_res[1], img_res[0]])
                else:
                    ground_truth = load_rgb_16bit(partial_path, [img_res[1], img_res[0]])

                gt_img[mask,:] = ground_truth.view([1,-1,3]).squeeze(dim=0).to(device)[mask,:]
                nvdiffrecmc_img[mask,:] = nvdiffrecmc.view([1,-1,3]).squeeze(dim=0).to(device)[mask,:]
                rendered_img = out_img[mask, :]
                ground_truth = gt_img[mask, :]
                nvdiffrecmc = nvdiffrecmc_img[mask, :]
                # calculating metrics
                metrics['tone_loss'].append(
                    ((tone_loss(rendered_img, ground_truth)) / float(points.shape[0])).detach().item())
                metrics['tone_loss_nv'].append(
                    ((tone_loss(nvdiffrecmc, ground_truth)) / float(points.shape[0])).detach().item())
                psnr_loss = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(
                    ((rendered_img - ground_truth) ** 2).mean(())).mean()
                psnr_loss_nv = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(
                    ((nvdiffrecmc - ground_truth) ** 2).mean(())).mean()
                ssim_loss = ssim(out_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0),
                                 gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0), data_range=1.0,
                                 size_average=False)
                ssim_loss_nv = ssim(nvdiffrecmc_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0),
                                 gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0), data_range=1.0,
                                 size_average=False)
                lpips_val = lpips_loss(
                    (-1 + (((out_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                             torch.min(out_img)) * 2) / (torch.max(out_img) - torch.min(out_img)))
                     ).to('cpu'),
                    (-1 + (((gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                             torch.min(gt_img)) * 2) / (torch.max(gt_img) - torch.min(gt_img)))).to(
                        'cpu'))
                lpips_val_nv = lpips_loss(
                    (-1 + (((nvdiffrecmc_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                             torch.min(nvdiffrecmc_img)) * 2) / (torch.max(nvdiffrecmc_img) - torch.min(nvdiffrecmc_img)))
                     ).to('cpu'),
                    (-1 + (((gt_img.view([img_res[1], img_res[0], 3]).permute(2, 0, 1).unsqueeze(0) -
                             torch.min(gt_img)) * 2) / (torch.max(gt_img) - torch.min(gt_img)))).to(
                        'cpu'))
                metrics['psnr'].append(psnr_loss.detach().item())
                metrics['psnr_nv'].append(psnr_loss_nv.detach().item())
                metrics['ssim'].append(ssim_loss.detach().item())
                metrics['ssim_nv'].append(ssim_loss_nv.detach().item())
                metrics['lpips'].append(lpips_val.detach().item())
                metrics['lpips_nv'].append(lpips_val_nv.detach().item())
                print(f"\n Camera {str(i)}, {envmap} envmap: tone loss = " + "{:.5f}".format(metrics['tone_loss'][-1])
                      + " -- PSNR(dB)= " + "{:.5f}".format(metrics['psnr'][-1])
                      + " -- SSIM= " + "{:.5f}".format(metrics['ssim'][-1])
                      + " -- LPIPS= " + "{:.5f}".format(metrics['lpips'][-1])
                        + " --nv PSNR(dB)= " + "{:.5f}".format(metrics['psnr_nv'][-1])
                        + " --nv SSIM= " + "{:.5f}".format(metrics['ssim_nv'][-1])
                        + " --nv LPIPS= " + "{:.5f}".format(metrics['lpips_nv'][-1]))

                out_img = torch.cat([nvdiffrecmc_img.view([img_res[1], img_res[0], 3]),out_img.view([img_res[1], img_res[0], 3]), gt_img.view([img_res[1], img_res[0], 3])], dim=1)
                pyredner.imwrite(out_img.cpu(), '{}/_RELIGHT/nvdiffrecmc_{}_val_{}_{}.png'.format(output_dir, cur_datetime, i,envmap))
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
        + " --nv PSNR(dB)= " + "{:.5f}".format(sum(metrics['psnr_nv']) / len(metrics['psnr_nv']))
        + " --nv SSIM= " + "{:.5f}".format(sum(metrics['ssim_nv']) / len(metrics['ssim_nv']))
        + " --nv LPIPS= " + "{:.5f}".format(sum(metrics['lpips_nv']) / len(metrics['lpips_nv'])))

    return

import torch
import mitsuba
import mitsuba as mi
import numpy as np
import os
import drjit as dr
from tqdm import tqdm


def render_buffer_nv(device, data_dir: str, output_dir: str, img_res: list):
    """
    For all training and validation cameras, it computes all the normals, light's output direction  and 3D position
    of every single point obtained by tracing rays on the surface of the object  from the camera, through all pixels.
    Every tensor computed, related to a specific camera, is then stored in a '.pt' (pytorch) file to be used as buffer
    and loaded during the training
    :param device:  device to load the tensors on
    :param data_dir: string, path to the data folder
    :param output_dir: string, path to the output folder
    :param img_res: list of ints, [width,height] resolution of the renderings
    :return: None
    """
    # loading camera parameters' files
    print("Computing buffer...")
    mitsuba.set_variant('cuda_ad_rgb')
    test_cameras = np.load(os.path.join(data_dir, 'cameras_val.npz'))

    Extri = np.array(test_cameras['Extri'])
    FOV = np.array(test_cameras['FOV'])

    mesh_path = output_dir + '/mesh/mesh.obj'

    n_images = test_cameras['Extri'].shape[0]
    total_cam_num = Extri.shape[0]
    test_cam_start = 0

    # creating buffer directory
    buffer_dir = data_dir + "/_nvBUFFER"
    if not os.path.exists(buffer_dir):
        os.makedirs(buffer_dir)

    # iterating through all cameras
    for c in tqdm(range(total_cam_num)):
        # rendering the normal, light's output direction  and 3D position of every
        # single point on the surface of the object seen from "c" camera's perspective
        pos_input, mask = make_scene(c, Extri, FOV, img_res, mesh_path, device)
        # saving normal, light's output direction  and 3D position in the buffer folder
        torch.save({'total_cam_num': total_cam_num,
                    'test_cam_start': test_cam_start,
                    'olat_cam_start': n_images,
                    'pos_input': pos_input,
                    'mask': mask,
                    }, '{}/{}.pt'.format(buffer_dir, c))
    print("Buffer computation finished")


def make_scene(index: int, extri, fov, img_res: list, mesh_path: str, device):
    """
    Given the 'index' camera, it computes all the normals, light's output direction  and 3D position of every
    single point obtained by tracing rays on the surface of the object  from "index" camera, through all pixels
    (given resolution 'img_res')
    :param index: int, index of the camera
    :param extri: extrinsic parameters of the camera
    :param fov: field of view of the camera
    :param img_res: list of ints, [width,height] resolution of the renderings
    :param mesh_path: string, path to the mesh of the object
    :param device: device to load the tensors on
    :return: pos_input, mask. The first one is a tensor of size [[img_res],9] containing points 3d positions (0:2),
     rays out directions (3:5) and normals (6:8). The second one is a bool tensor of img_res size, masking out the
     background from the object
    """
    # creating the rotation matrix R_m from extrinsic parameters to match Mitsuba's convention
    rotation = extri[index]
    fov = fov[index]
    rotation[:, 1] *= -1
    rotation[:, 0] *= -1
    rotation_list = rotation.reshape(-1).tolist()
    r_m = " ".join(str(x) for x in rotation_list)
    # creating the scene
    pos_scene = mi.load_string("""
        <?xml version="1.0"?>
        <scene version="3.0.0">
            <integrator type="aov">
                <string name="aovs" value="pos:position,nn:sh_normal"/>
                <integrator type="path" name="my_image"/>
            </integrator>


            <sensor type="perspective">
                <transform name="to_world">
                     <matrix value="{matrix_values}"/>

                     </transform>
                    <float name="fov" value="{fov}"/>   

                <sampler type="independent">
                    <integer name="sample_count" value="1"/>

                </sampler>

                <film type="hdrfilm">
                    <integer name="width" value="{W}"/>
                    <integer name="height" value="{H}"/>
                    <rfilter type="box"/>
                </film>
            </sensor>

            <shape type="obj">

                <string name="filename" value="{mesh_path}"/>   

            </shape>

        </scene>
    """.format(matrix_values=r_m, fov=fov, W=img_res[0], H=img_res[1], mesh_path=mesh_path))

    pos_params = mi.traverse(pos_scene)
    # rotating the mesh of 90 degrees in the X-axis
    ver = dr.unravel(mi.Point3f, pos_params['OBJMesh.vertex_positions'])
    t = mi.Transform4f.rotate(axis=[1, 0, 0], angle=90)
    rot_ver = t @ ver
    pos_params['OBJMesh.vertex_positions'] = dr.ravel(rot_ver)
    pos_params.update()
    # rendering the normals, light's output directions  and 3D positions
    rendering = mi.render(pos_scene, params=pos_params, spp=1)
    pos_img = torch.tensor(rendering, device=device).detach()

    mask = pos_img[:, :, -1].float().view(-1) > 0
    positions = pos_img[:, :, 3:6].view(-1, 3)[mask, :]
    normals = pos_img[:, :, 6:9].view(-1, 3)[mask, :]

    cam_loc = torch.tensor(rotation[:3, 3]).to(device).view([1, 3])
    ray_dir = (cam_loc - positions)
    ray_dir_normed = ray_dir / (ray_dir.norm(2, dim=1).unsqueeze(-1))

    pos_input = torch.cat([positions, ray_dir_normed, normals], dim=-1)

    return pos_input, mask



