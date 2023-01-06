import os
from pathlib import Path
import torch

from src.relight_w_nvdiffrecmc import relight_w_nvdiffrecmc, render_buffer_nv


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
    #render_buffer_nv(device=device, data_dir=data_dir, output_dir=output_dir, img_res=img_res)

    joint_train_datetime = config("model_datetime")
    relight_w_nvdiffrecmc(output_dir, data_dir, device, img_res, joint_train_datetime, config['num_views'],
                          config['num_envmaps'],
                          config['mlp_size'], scene_name=config['scene_name'])
    return


if __name__ == '__main__':
    config = {
        "scene_name": 'nerfactor_ficus',
        "data_dir": None, # path to dataset folder. If None, it is set to <project's root>\data\<scene_name>
        "output_dir": None,  # path to nvdiffrecmc's output folder. If None, it is set to <project's root>\out\<scene_name>
        "device": 'cuda:1',  # options: auto, cpu, cuda:<id gpu>
        "res_w": 512,  # desired resolution (width) for training images (down/up-scaling is built-in)
        "res_h": 512,  # desired resolution (height) for training images (down/up-scaling is built-in)
        "n_olat_val": 100,  # number of OLAT training images to validate the training with
        "olat_training_batch": 0,  # number of pixels per OLAT image to train the model with in every iteration. If set to 0, the whole image will be used
        "olat_iters": 100000,  # number of training iterations for OLAT training
        "rgb_training_batch": 500,  # number of pixels per rgb (real captures) image to train the model with in every iteration
        "joint_iters": 50000,  # number of training iterations for joint training
        "log_time_mem": True,  # if True, the time and memory consumption of every step will be logged
        "num_views": 8, # number of views to choose randomly among available to render relighting images
        "num_envmaps": 3,  # number of envmaps to choose randomly among available to render relighting images
        "mlp_size":[128,7,3],  # size of the MLP used in the model. First int is the number of neurons for every layer,
        # the second int is the number of layers, the third int is the number of the layer to place a skip connection
        "lr_envmap": 2e-3,  # learning rate for the envmap
        "model_datetime": "2022-12-31_17-47"  # datetime of the model to be tested
        }
    main(config)




