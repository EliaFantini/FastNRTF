import os
import torch
from torchvision.io import read_image
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
from skimage.io import imsave, imread
from tqdm import tqdm

from src.hash_encoding import SHEncoder, HashEmbedder


def load_rgb(path: str,target_res: list):
    """
    Loads a rgb image from disk and returns it as a tensor.
    The image's values are automatically set to float in the range [0,1].
    If target_res is specified, it is automatically
    up/down-scaled to the required resolution.

    :param path: str, path where to load the image
    :param target_res: list of [int,int], optional. The required resolution of the loaded image
    :return: tensor, the loaded rgb image
    """
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    envmap_img = read_image(files[0])[:3,:,:]
    img = envmap_img.unsqueeze(0).to(torch.float) # CHW -> NCHW
    img = torch.nn.functional.interpolate(img, size = target_res, mode = 'area')
    img = img.squeeze(dim=0)  # NCHW -> CHW
    img = img.permute(1, 2, 0)/255  #CHW->HWC
    envmap_img_lowres = img
  
    return envmap_img_lowres


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    """
    Converts an image (passed as a tensor) from sRGB color space to RGB.
    :param f: tensor, sRGBimage
    :return: tensor, RGB image converted
    """
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))


def load_rgb_16bit(path: str, target_res: list = None, device = None):
    """
    Loads a rgb image from disk encoded with 16 bit precision and returns it as a tensor.
    The image's values are automatically set to float in the range [0,1]. The image is also
    converted from sRGB color space to rgb and if target_res is specified, it is automatically
    up/down-scaled to the required resolution.

    :param path: str, path where to load the image
    :param target_res: list of [int,int], optional. The required resolution of the loaded image
    :param device: device where to load the image on, as a tensor
    :return: tensor, the loaded rgb image
    """
    # loading the image
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    image = skimage.io.imread(files[0])[:,:,:3]
    image = skimage.img_as_float(image).astype(np.float32)
    image = torch.from_numpy(image).float()
    # converting sRGB to rgb
    img = torch.cat((_srgb_to_rgb(image[..., 0:3]), image[..., 3:4]), dim=-1) if image.shape[-1] == 4 else _srgb_to_rgb(image)
    if target_res is not None and (target_res[0]!=img.shape[0] or target_res[1]!=img.shape[1]):
        # up/down-scaling the image
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0).to(torch.float)  # CHW -> NCHW
        img = torch.nn.functional.interpolate(img, size=target_res, mode='area')
        img = img.squeeze(dim=0)  # NCHW -> CHW
        img = img.permute(1, 2, 0) # CHW-> HWC
    img_lowres = img
    if device is not None:
        # moving the image's tensor to device
        img_lowres = img_lowres.to(device)
    return img_lowres


def ewma(data, window=5):
    """
    Exponentially-weighted moving average.

    :param data: list-like
        The data to be averaged.
    :param window: int, optional
        The length of the averaging window.
    :return: ndarray
        The exponentially-weighted moving average of the data.
    """

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def plot(data: dict, path: str, scale: str, x_values: list):
    """
    This function plots the data in a dictionary.

    :param data: dict,
        The dictionary should have the following structure:
        {
            'key1': [value1, value2, ...],
            'key2': [value1, value2, ...],
            ...
        }
        The keys will be used as labels for the plots and values will be plotted against the x_values.
    :param path: str, path where to save the output file with the plots
    :param scale: str, vertical scale used to plot values. The scale can be 'linear' or 'log'
    :param x_values: list, values on the x-axis
    :return: None
    """
    keys = data.keys()
    fig, axs = plt.subplots(int(len(keys)/3)+1, 3)
    # creating a plot for every list in the dictionary
    for i, key in enumerate(keys):
        # plotting the values in the list against the x_values
        axs.flat[i].plot(x_values, ewma(np.array(data[key])), label= str(key))
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
    fig.text(0.5, 0.04, 'Iterations', ha='center')
    fig.text(0.04, 0.5, 'Metrics', va='center', rotation='vertical')
    fig = plt.gcf()
    fig.set_size_inches(5*len(keys), 5)
    fig.savefig(path)
    plt.close(fig)


def get_hashembedder(bounding_box: list = [-1, 1], vertices=None, device=None):
    """
    This function returns a hash embedder object and the output dimension of the hash embedder.

    :param bounding_box: list, optional
        The bounding box of the point cloud.
    :param vertices: list, optional
        The vertices of the point cloud.
    :param device:  str, optional
        The device to use.
    :return:
        embed : HashEmbedder,
            The hash embedder object.
        out_dim : int,
            The output dimension of the hash embedder.
    """
    embed = HashEmbedder(bounding_box=bounding_box,
                         n_features_per_level=16,
                         base_resolution=16.,
                         n_levels=19,
                         finest_resolution=256.,
                         sparse=True,
                         vertices=vertices,
                         device=device
                         )
    out_dim = embed.out_dim

    return embed, out_dim


def get_SHembedder():
    """
    Initializes the spherical harmonics embedder and returns its instance with the output dimension as an int
    :return:  SHEncoder instance, output dimension as an int
    """
    embed = SHEncoder()
    out_dim = embed.out_dim

    return embed, out_dim


def uv_to_dir(u: int,v: int, envmap_height: int = 16):
    """
    Convert uv coordinates to a direction vector.

    :param u:int
        The u coordinate of the pixel.
    :param v: int
        The v coordinate of the pixel.
    :param envmap_height: int
        The height of the environment map.
    :return:
        x : float
            The x component of the direction vector.
        y : float
            The y component of the direction vector.
        z : float
            The z component of the direction vector.

    """
    theta=(u+0.5)*np.pi/envmap_height
    phi=(v+0.5)*2*np.pi/(envmap_height*2)
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)


def image_to_pt(data_dir: str):
    """
    Converts every png image (generated OLAT images) into a pytorch tensor and saving it, after applying srgb_to_rgb
    conversion to them
    :param data_dir: string, path to the data folder
    :return: None
    """
    data_dir = data_dir + "/_OLAT"
    print("Converting images to pytorch's tensors...")
    for i, filename in tqdm(enumerate(glob.glob(data_dir + '/*.png'))):
        olat_file = filename[:-4] + ".pt"
        if os.path.exists(filename):
            olat = load_rgb_16bit(filename[:-4])
            torch.save(olat, olat_file)
            # os.remove(filename)
    print("Finished conversion")


def rotate_envmap(output_dir: str):
    """
    Rotates the nvdiffrecmc's estimated envmap by -90 degrees, after downscaling it to the size [32,16]
    :param output_dir: string, path to the output folder
    :return: None
    """
    print("Rotating estimated envmap by -90°")
    probe_path = output_dir + '/mesh/probe.hdr'
    envmap = imread(probe_path)
    envmap = envmap[:, :, :3]
    # downscaling the envmap to the resolution of [32,16]
    envmap = cv2.resize(envmap, [32, 16], interpolation=cv2.INTER_LINEAR)
    M = np.float32([[1, 0, -8], [0, 1, 0]])
    (rows, cols) = envmap.shape[:2]
    envmap_backup = envmap
    # rotating the envmap by -90 degrees
    envmap = cv2.warpAffine(envmap, M, (cols, rows))
    envmap[:, -8:, :] = envmap_backup[:, :8, :]
    # saving the modified envmap
    imsave(probe_path[:-4] + '_rot' + '.hdr', envmap)


def tone_loss(x,y):
    """
    This function takes two tensors as input and returns the tone loss between them.
    The tone loss is defined as the L1.8 norm of the difference between the two tensors,
    divided by the first tensor (x) + eps=0.001, without gradient computation on the divider.

    @param x: torch.Tensor, The first input tensor.
    @param y: torch.Tensor, The second input tensor.
    @return: torch.Tensor, The tone loss between the two input tensors.
    """
    dif=x-y
    mapping=x.detach().clamp(1e-3,1)
    loss=dif/mapping
    return torch.norm(loss,p=1.8)
