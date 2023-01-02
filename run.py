import os
from pathlib import Path
import torch

from src.relight import relight
from src.render_buffer import render_buffer
from src.train_joint import train_joint
from src.train_olat import train_olat
from src.utils import image_to_pt, rotate_envmap


def main(config: dict):
    """
    Main function of the program.
    It loads the configuration's settings, renders the buffer containing  the normals, light's output direction
    and 3D position of points on the object's surface, turns every png image (generated OLAT images) into a pytorch
    tensor and saves it, rotates the nvdiffrecmc's estimated envmap by -90 degrees, trains the model with generated
    OLAT images, trains the model both on OLAT images and on real captures, optimizing the estimated envmap too,
    tests model by relighting it and comparing it with ground truth images.

    :param config: dict , dictionary containing the configuration
    """
    # loading configuration's settings
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
    data_dir = root_dir + f"/data/{config['scene_name']}"
    if config['data_dir'] is not None:
        data_dir = config['data_dir']

    output_dir = root_dir + f"/out/{config['scene_name']}"
    if config['output_dir'] is not None:
        output_dir = config['output_dir']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config['device'] != 'auto':
        device = torch.device(config['device'])
    img_res = [config['res_w'], config['res_h']]
    n_olat_val = config['n_olat_val']
    olat_training_batch = config['olat_training_batch']
    rgb_training_batch = config['rgb_training_batch']
    olat_iters = config['olat_iters']
    joint_iters = config['joint_iters']

    print(f"Config:\n   scene name -- {config['scene_name']}\n  data dir -- {data_dir}\n    output dir -- {output_dir}"
          f"\n  device -- {device}\n    image res -- {img_res}\n    n olat val -- {n_olat_val}\n"
          f"    olat training batch -- {olat_training_batch}\n  olat iters -- {olat_iters}\n    rgb training batch -- "
          f"{rgb_training_batch}\n  joint_iters -- {joint_iters}\n  log_time_mem -- {config['log_time_mem']}"
          f"\n  num_views -- {config['num_views']}\n  num_envmaps -- {config['num_envmaps']}"
          f"\n  mlp_size -- {config['mlp_size']}\n  lr_envmap -- {config['lr_envmap']}")

    # rendering the buffer containing  the normals, light's output direction
    # and 3D position of points on the object's surface
    render_buffer(device=device, data_dir=data_dir, output_dir= output_dir, img_res= img_res)
    # turning every png image (generated OLAT images) into a pytorch tensor and saving it
    image_to_pt(data_dir)
    # rotating the nvdiffrecmc's estimated envmap by -90 degrees
    rotate_envmap(output_dir)
    # training the model with generated OLAT images
    olat_train_datetime = train_olat(output_dir, data_dir, device, n_olat_val, img_res, olat_training_batch, olat_iters, config['log_time_mem'], config['mlp_size'])
    # training the model both on OLAT images and on real captures, optimizing the estimated envmap too
    joint_train_datetime = train_joint(output_dir, data_dir, device, img_res, rgb_training_batch, olat_training_batch,
                                       olat_model_name=olat_train_datetime, joint_iterations=joint_iters, log_time_mem=config['log_time_mem'], mlp_size=config['mlp_size'], lr_env=config['lr_envmap'])
    # testing model by relighting it and comparing it with ground truth images
    relight(output_dir, data_dir, device, img_res, joint_train_datetime, config['num_views'], config['num_envmaps'], config['mlp_size'])

    return


if __name__ == '__main__':
    config = {
        "scene_name": 'nerfactor_ficus',
        "data_dir": None, # path to dataset folder. If None, it is set to <project's root>\data\<scene_name>
        "output_dir": None,  # path to nvdiffrecmc's output folder. If None, it is set to <project's root>\out\<scene_name>
        "device": 'cuda:0',  # options: auto, cpu, cuda:<id gpu>
        "res_w": 512,  # desired resolution (width) for training images (down/up-scaling is built-in)
        "res_h": 512,  # desired resolution (height) for training images (down/up-scaling is built-in)
        "n_olat_val": 100,  # number of OLAT training images to validate the training with
        "olat_training_batch": 0,  # number of pixels per OLAT image to train the model with in every iteration. If set to 0, the whole image will be used
        "olat_iters": 20000,  # number of training iterations for OLAT training
        "rgb_training_batch": 250,  # number of pixels per rgb (real captures) image to train the model with in every iteration
        "joint_iters": 10000,  # number of training iterations for joint training
        "log_time_mem": True,  # if True, the time and memory consumption of every step will be logged
        "num_views": 8, # number of views to choose randomly among available to render relighting images
        "num_envmaps": 3,  # number of envmaps to choose randomly among available to render relighting images
        "mlp_size":[512,7,3],  # size of the MLP used in the model. First int is the number of neurons for every layer,
        # the second int is the number of layers, the third int is the number of the layer to place a skip connection
        "lr_envmap": 2e-2,  # learning rate for the envmap
    }
    main(config)




