<p align="center">
  <img alt="FastNRTF" src="https://user-images.githubusercontent.com/62103572/210257218-6bef5f4a-aa91-41ec-986c-2b90e02a0a1a.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/FastNRTF">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/FastNRTF">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/FastNRTF">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/FastNRTF">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/FastNRTF?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/FastNRTF?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/FastNRTF?style=social">
</p>

FastNRTF is an EPFL's MSc Data Science Semester Project aiming to optimize memory and time requirements in the process of inverse rendering for reconstructing 3D scenes from a set of images. This involves estimating various properties of
objects, such as material, lighting, and geometry. While recent approaches have
enabled the relighting of scenes under novel illumination conditions, they often rely
on assumptions, such as direct illumination and predetermined material models,
which can lead to an incomplete reconstruction of interreflections and shadow-free
albedos. To address this issue, we utilize the neural precomputed radiance transfer
function proposed in [Neural Radiance Transfer Field](https://github.com/LinjieLyu/NRTF) paper to handle complex global illumination effects. However,
the computation required for this method is intensive and requires a powerful GPU
with a large amount of memory, as well as a significant amount of time to train the
model. Therefore, we employ Monte Carlo path tracing and denoising from [Nvdiffrecmc](https://github.com/NVlabs/nvdiffrecmc)
for initial shape, light, and material reconstruction, and integrate it into the training
framework to optimize for time and memory consumption. Our approach
results in a roughly 10x reduction in training time and a minimum required VRAM
of 6GB, while still producing high-quality relighting renderings.

The code is based on the [Neural Radiance Transfer Field](https://github.com/LinjieLyu/NRTF) framework and contains some functions and code snippets borrowed from their repository.  Their code serves as the foundation for the present implementation. Additionally, a portion of the code from [Nvdiffrecmc](https://github.com/NVlabs/nvdiffrecmc) has been incorporated in order to correctly load their material model into Blender.

## Author

- [Elia Fantini](https://github.com/EliaFantini/)

## How to install
Requires Python 3.6+, VS2019+, Cuda 11.3+ and PyTorch 1.10+, and an NVIDIA GPU with a modern driver supporting OptiX 7.3 or newer.

Tested in Anaconda3 with Python 3.8 and PyTorch 1.13.1 on the following GPUs: RTX 3070, GTX Titan X.

1. Download this repository as a zip file and extract it into a folder. 
2. Download code from the [official implementation](https://github.com/NVlabs/nvdiffrecmc) of Nvdiffrecmc as a zip file and extract it into another folder.
2. Install Blender (3.1.2 tested).

4. The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/

Then from an Anaconda prompt use the following commands
```
conda create -n nrtf python=3.8
conda activate nrtf
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
imageio_download_bin freeimage
pip install tqdm scikit-image opencv-python pandas tensorboard addict imageio imageio-ffmpeg pyquaternion scikit-learn pyyaml seaborn PyMCubes trimesh plyfile redner-gpu matplotlib jupyter lpips pytorch-msssim mitsuba
```
5. Then run the following python script in the 'src' folder:
```
python generate_olat_envmaps.py
```
This code has to be run only once. 
6. Download [NeRFactor](https://xiuming.info/projects/nerfactor/) dataset. Our experiments were run on this dataset and Nvdiffrecmc's repository provides a [script](https://github.com/NVlabs/nvdiffrecmc/blob/main/data/download_datasets.py) to download it easily. Comment the last lines "download_nerf_synthetic()"
"download_nerd()" before running it to avoid downloading unnecessary data. Then run:
```
python download_datasets.py
```

## How to use
After installation, to reproduce results for a scene do the following (we use NeRFactor's Ficus scene as an example):
1. Following [Nvdiffrecmc's repository](https://github.com/NVlabs/nvdiffrecmc) section "Examples", run the script:
```
python train.py --config configs/nerfactor_ficus.json
```
If you get OUT_OF_MEMORY exception from PyTorch you should reduce the 'batch' variable in the 'configs/nerfactor_ficus.json' file. In our experiments with GPU having equal or less than 12GB of VRAM we used batch size 1. To improve results we used batch size to 6 and added the line of code "FLAGS.__dict__['batch'] = 1" right after line 646 in their [train function](https://github.com/NVlabs/nvdiffrecmc/blob/main/train.py).

Once the training it's done you can specify a personalized "data_dir" in the **config** dictionary in the `run.py` script that contains  ficus' data and a personalized "output_dir" that contains a `mesh` folder containing the files generated by Nvdiffrecmc (mesh + materials + envmap). Otherwise you can use default settings and copy the folder containing ficus' data (`ficus_2188` if downloaded using Nvdiffrecmc's script to download NeRFactor's dataset) in our `data` folder. Then in the main folder create a new folder `out` and inside it create a folder with the same name as the folder pasted in `data`, in this case it will be `ficus_2188`. Finally copy the `mesh` folder created by nvdiffrecmc containing the estimated mesh, materials and envmap inside `out/ficus_2188`. The file structure will be the following
```
<project_path>/data/<scene_name>/
    test_000
    test_001
    ...
    train_000
    ...
    val000
    ...
    transforms_test.json
    transforms_train.json
    transforms_val.json
    
<project_path>/dout/<scene_name>/mesh/
    mesh.obj
    probe.hdr
    texture_kd.png
    ...
```
2. Open an empty scene in Blender, go into the scripting mode and open the code `blender/render_OLAT_images.py`, set PROJECTS_FOLDER to your project path and SCENE_NAME to <scene_name> (in our example it's 'ficus_2188'). Then run the script.
3. Specify your preferred settings in the *config* dictionary inside the `run.py` script (or leave them unchanged for default settings) then run:
```
python run.py
```
## Training settings
The **config** dictionary contains the following variables to be optionally changed:
```
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
```
## Results
Example of results of our experiments on the four NeRFactor's scenes
![nvdiffrecmc_ours_reference](https://user-images.githubusercontent.com/62103572/210345045-a85455c9-f24b-4d30-92a4-43a2d8b0d272.png)


## Files description
- **blender/**: folder containing the python file to be opened and run in Blender.
- **data/**: folder containing the training images divided in subfolders named after the scene they contain images of.
- **data/light_probes/**: folder containing [NeRFactor](https://xiuming.info/projects/nerfactor/)'s test envmaps to do relighting.
- **experiments/**: folder containing the code used to analyse the metrics obtained from different experiments and to generate the figures and plot that were put into the report. It also contains code used to compare our relight with reference and with the initial estimations obtained with Nvdiffrecmc. Finally, it contains the main function of Nvdiffrecmc "train.py" modified to log time-memory consumptions and with a different batch size for phase 2. More details are available in readme files inside the folders.
- **figures/**: folder containing the figures for the report.
- **src/**: folder containing most of the functions and helpers to run the training.
- **run.py**: main file, containing the main function for the whole training process after the Nvdiffrecmc's initial estimation.








