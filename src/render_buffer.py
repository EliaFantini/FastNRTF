import torch
import mitsuba
import mitsuba as mi
import numpy as np
import os
import drjit as dr
from tqdm import tqdm


def render_buffer(device, data_dir: str, output_dir: str, img_res: list):
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
    input_cameras = np.load(os.path.join(data_dir, 'cameras.npz'))
    olat_cameras = np.load(os.path.join(data_dir, 'cameras_olat.npz'))
    test_cameras = np.load(os.path.join(data_dir, 'cameras_test.npz'))

    Extri = np.concatenate((input_cameras['Extri'], olat_cameras['Extri'], test_cameras['Extri']), axis=0)
    FOV = np.concatenate((input_cameras['FOV'], olat_cameras['FOV'], test_cameras['FOV']), axis=0)

    mesh_path = output_dir + '/mesh/mesh.obj'

    n_images = input_cameras['Extri'].shape[0]
    total_cam_num = Extri.shape[0]
    test_cam_start = input_cameras['Extri'].shape[0] + olat_cameras['Extri'].shape[0]

    # creating buffer directory
    buffer_dir = data_dir + "/_BUFFER"
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
