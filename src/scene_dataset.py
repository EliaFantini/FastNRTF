import os
import torch
import json
import src.utils as utils

class SceneDataset(torch.utils.data.Dataset):
    """
    This class is a torch.utils.data.Dataset object that loads the data (images) from the dataset.
    It is used to load the data for training and validation.

    Args:
        transform_path (str): Path to the json file that contains the information about the dataset.
        img_res (list): list of int [height, width]. The desired resolution of the images.Images will be up/down-scaled to this resolution
        nerfactor (bool): If True, the images are loaded as 16 bit images and converted from sRGB to RGB.

    Attributes:
        n_images (int): The number of images in the dataset.
        rgb_images (tensor): Tensor containing all images loaded
    """
    def __init__(self,
                 transform_path: str,
                 img_res: list,
                 nerfactor: bool = False
                 ):
        # loading the dictionary that contains the information about the dataset (files' paths).
        cfg = json.load(open(transform_path, 'r'))
        self.n_images = len(cfg['frames'])

        main_dir = os.path.dirname(transform_path)

        print("Loading dataset...")

        assert os.path.exists(main_dir), "Data directory is empty"
        
        self.rgb_images = torch.zeros([self.n_images,img_res[0],img_res[1],3],  dtype=torch.float32)
        # loading images
        for i in range(self.n_images):
            if nerfactor:
                self.rgb_images[i, :, :, :] = utils.load_rgb_16bit(os.path.join(main_dir, cfg['frames'][i]['file_path']), img_res)
            else:
                self.rgb_images[i, :, :, :] = utils.load_rgb(os.path.join(main_dir, cfg['frames'][i]['file_path']), img_res)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        return idx, self.rgb_images[idx]

   
