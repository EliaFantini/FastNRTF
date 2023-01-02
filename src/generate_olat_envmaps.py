import os
from pathlib import Path
import torch
import pyredner


def generate_olat_envmaps():
    """
    This function generates and saves the OLAT envmaps in the data folder inside the root folder of the project.
    :return: None
    """
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute())

    # creating the output directory
    env_dir = root_dir + r"\data\OLAT_envmaps"
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    # generating and saving the envmaps
    for i in range(0, 512):
        img = torch.zeros([512, 3])
        img[i, :] = 100
        pyredner.imwrite(img.view(16, 32, 3).cpu(), '{}/{}.exr'.format(env_dir, i), gamma=1)


if __name__ == '__main__':
    generate_olat_envmaps()
