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



## Authors

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


## Files description
- **blender/**: folder containing the python file to be opened and run in Blender.
- **data/**: folder containing the training images divided in subfolders named after the scene they contain images of.
- **data/light_probes/**: folder containing [NeRFactor](https://xiuming.info/projects/nerfactor/)'s test envmaps to do relighting.
- **figures/**: folder containing the figures for the report.
- **src/**: folder containing most of the functions and helpers to run the training.
- **run.py**: main file, containing the main function for the whole training process after the Nvdiffrecmc's initial estimation.







