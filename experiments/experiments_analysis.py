import pickle
from pathlib import Path
import torch
import pyredner

from experiments_utils import *

############################################################################################
# calculating the scene with the highest memory consumption in JOINT optimization
############################################################################################

# loading the metrics of the JOINT optimization
root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
scene_name = "nerfactor_hotdog"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-26_10-23.pkl', 'rb') as f:
    hotdog = pickle.load(f)
scene_name = "nerfactor_ficus"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-28_10-11.pkl', 'rb') as f:
    ficus = pickle.load(f)
scene_name = "nerfactor_lego"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-28_10-19.pkl', 'rb') as f:
    lego = pickle.load(f)
scene_name = "nerfactor_drums"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-29_19-18.pkl', 'rb') as f:
    drums = pickle.load(f)

# print max memory usage of each scene
print("\nMAX MEMORY USAGE per scene:")
print("-Lego:" + str(np.array(lego['max_all_mem']).max()) + "GB")
print("-Drums:" + str(np.array(drums['max_all_mem']).max())+ "GB")
print("-Hotdog:" + str(np.array(hotdog['max_all_mem']).max())+ "GB")
print("-Ficus:" + str(np.array(ficus['max_all_mem']).max())+ "GB")


############################################################################################
# calculating max memory and plotting metrics for hotdog scene for different rgb batch sizes
############################################################################################

# loading the metrics of the experiments
scene_name = "nerfactor_hotdog"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-26_10-23.pkl', 'rb') as f:
    hotdog_rgb250 = pickle.load(f)
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-27_09-08.pkl', 'rb') as f:
    hotdog_rgb500 = pickle.load(f)

# printin the max memory usage of the experiments and their number of iterations
print("\nJOINT optimization batch size experiment: MAX MEMORY USAGE - #ITERATIONS:")
print("-Lego batch size 250:" + str(np.array(hotdog_rgb250['max_all_mem']).max())+ "GB" + " - "+str(len(hotdog_rgb250['max_all_mem'])) + " iterations")
print("-Lego batch size 500:" + str(np.array(hotdog_rgb500['max_all_mem']).max())+ "GB" + " - "+str(len(hotdog_rgb500['max_all_mem'])) + " iterations")
hotdog_rgb250 = keep_only_metrics_no_memory_log(hotdog_rgb250)
hotdog_rgb500 = keep_only_metrics_no_memory_log(hotdog_rgb500)


plot_rgb_batch_exp(hotdog_rgb250, hotdog_rgb500, "linear")

############################################################################################################
# Experiment Lego scene with different numbers of training images for OLAT Training
############################################################################################################
# loading the metrics of the experiments
scene_name = "nerfactor_lego"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-26_21-00.pkl', 'rb') as f:
    lego_1650 = pickle.load(f)
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-26_20-53.pkl', 'rb') as f:
    lego_1100 = pickle.load(f)
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-14_21-34.pkl', 'rb') as f:
    lego_2100 = pickle.load(f)

lego_1650 = keep_only_metrics_no_memory_log(lego_1650)
lego_1100 = keep_only_metrics_no_memory_log(lego_1100)
lego_2100 = keep_only_metrics_no_memory_log(lego_2100)


plot_OLAT_num_train_images_exp(lego_1100, lego_1650, lego_2100, "linear")


############################################################################################################
# plotting the metrics of olat training with 3 different numbers of olat batch sizes
############################################################################################################
# loading the metrics of the experiments
scene_name = "nerfactor_lego"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-27_08-46.pkl', 'rb') as f:
    lego_10k = pickle.load(f)
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-27_09-52.pkl', 'rb') as f:
    lego_500 = pickle.load(f)
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-14_21-34.pkl', 'rb') as f:
    lego_full = pickle.load(f)

lego_10k = keep_only_metrics_no_memory_log(lego_10k)
lego_500 = keep_only_metrics_no_memory_log(lego_500)
lego_full = keep_only_metrics_no_memory_log(lego_full)


plot_OLAT_batch_size_exp(lego_500, lego_10k, lego_full, "linear")



#######################################################################################################################
# Experiment with different learning rates for the envmap in Lego scene
#######################################################################################################################
# load the data
scene_name = "nerfactor_lego"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-27_19-51.pkl', 'rb') as f:
    lego_env2e2 = pickle.load(f)
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-27_20-25.pkl', 'rb') as f:
    lego_env2e1 = pickle.load(f)
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-28_10-19.pkl', 'rb') as f:
    lego_env2e3 = pickle.load(f)
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-30_21-53.pkl', 'rb') as f:
    lego_env1 = pickle.load(f)

print("\nExperiment with different learning rates for the envmap in Lego scene: max memory usage - #iterations")
print("- lr = 0.002:" + str(np.array(lego_env2e3['max_all_mem']).max()) + " GB - "+str(len(lego_env2e3['max_all_mem'])) + " iterations")
print("- lr = 0.02:" + str(np.array(lego_env2e2['max_all_mem']).max()) + " GB - "+str(len(lego_env2e2['max_all_mem'])) + " iterations")
print("- lr = 0.2:" + str(np.array(lego_env2e1['max_all_mem']).max()) +" GB - "+str(len(lego_env2e1['max_all_mem'])) + " iterations")
print("- lr = 1:" + str(np.array(lego_env1['max_all_mem']).max()) + " GB - "+str(len(lego_env1['max_all_mem'])) + " iterations")
lego_env2e1 = keep_only_metrics_no_memory_log(lego_env2e1)
lego_env2e2 = keep_only_metrics_no_memory_log(lego_env2e2)
lego_env2e3 = keep_only_metrics_no_memory_log(lego_env2e3)
lego_env1 = keep_only_metrics_no_memory_log(lego_env1)

plot_envmap_lr_exp( lego_env1,lego_env2e1,lego_env2e2, "linear")


########################################################################################################################
# Generating image with comparison of different estimated envmaps and reference
########################################################################################################################
# load the data
scene_name = "nerfactor_lego"
out_dir = root_dir + f"/out/{scene_name}"
ckpt = torch.load('{}/_JOINT/checkpoint/2022-12-30_21-53_latest.pt'.format(out_dir))
nrtf1_envmap = ckpt['Envmaps'].to('cuda')
nrtf_envmap = pyredner.imread('{}/_JOINT/2022-12-27_19-51_env_16300.png'.format(out_dir), gamma=1).to('cuda')
envmap_nvdiffrecmc = pyredner.imread('{}/mesh/probe_rot.hdr'.format(out_dir), gamma=1).to('cuda')
envmap_reference = pyredner.imread(root_dir+"/data/nerfactor_lego/light_probes/2163.hdr", gamma=1).to('cuda')
# generate the image
psnr_envmap_nvdiffrecmc = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(
                    ((envmap_nvdiffrecmc - envmap_reference) ** 2).mean(())).mean()
psnr_envmap_nrtf = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(
                    ((nrtf_envmap - envmap_reference) ** 2).mean(())).mean()
psnr_envmap_nrtf1 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(((nrtf1_envmap - envmap_reference) ** 2).mean(())).mean()
img = torch.cat([envmap_nvdiffrecmc,nrtf_envmap,nrtf1_envmap, envmap_reference], dim=1)
pyredner.imwrite(img.cpu(), root_dir + "/figures/envmaps_comparison_lego.png")
# print the PSNR
print(f"\nPsnr_envmap_nvdiffrecmc: {psnr_envmap_nvdiffrecmc}dB - psnr_envmap_nrtf: {psnr_envmap_nrtf}dB - psnr_envmap_nrtf1: {psnr_envmap_nrtf1}dB")

########################################################################################################################
# calculating time spent for 100k iteration of OLAT training and 50k of JOINT training in hotdog scene
########################################################################################################################
# loading the metrics of the experiments
scene_name = "nerfactor_hotdog"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-25_17-06.pkl', 'rb') as f:
    hotdog_olat100k = pickle.load(f)
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-26_10-23.pkl', 'rb') as f:
    hotdog_joint50k = pickle.load(f)
print("\n100k OLAT + 50k JOINT experiment: TIME SPENT - #ITERATIONS:")
print("-Hotdog 100k iterations olat:" + str(sum(hotdog_olat100k['iter_time'])/60) + "mn - "+str(len(hotdog_olat100k['iter_time'])) + " iterations")
print("-Hotdog 50k iterations joint:" + str(sum(hotdog_joint50k['iter_time'])/60) + "mn - "+str(len(hotdog_joint50k['iter_time']))   + " iterations")

hotdog_olat100k = keep_only_metrics_no_memory_log(hotdog_olat100k)
hotdog_joint50k = keep_only_metrics_no_memory_log(hotdog_joint50k)

plot_100kOLAT_50kJOINT_exp( hotdog_olat100k,hotdog_joint50k, "linear")


########################################################################################################################
# Experiment with no OLAT Training only JOINT training on Hotdog scene, plotting the metrics and printing the time spent
########################################################################################################################
# loading the metrics of the experiments
scene_name = "nerfactor_hotdog"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-26_20-41.pkl', 'rb') as f:
    hotdog_onlyjoint = pickle.load(f)
print("\nNo OLAT Training only JOINT training experiment: TIME SPENT - #ITERATIONS:")
print("-Hotdog joint only:" + str(sum(hotdog_onlyjoint['iter_time'])/60) + "mn  -"+str(len(hotdog_onlyjoint['iter_time'])) + " iterations")

hotdog_onlyjoint = keep_only_metrics_no_memory_log(hotdog_onlyjoint)


plot_noOLAT_onlyJOINT_exp(hotdog_onlyjoint,  "linear")

############################################################################################################
# Printing memory usage for final experiment with Ficus small mlp, less OLAT training images,
# less iterations, higher envmap learning rate
############################################################################################################

# get the memory usage of the final experiment
scene_name = "nerfactor_ficus"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-31_17-47.pkl', 'rb') as f:
    ficus_final_exp = pickle.load(f)
print("\nMemory usage of final experiment with Ficus small MLP, less OLAT training images, less iterations, higher envmap learning rate:")
print("-Ficus_final_exp:" + str(np.array(ficus_final_exp['max_all_mem']).max()) + "GB - "+str(len(ficus_final_exp['max_all_mem'])) + " iterations")

############################################################################################################
# calculate metrics' average over all 4 scenes
############################################################################################################
# metrics from report's tab to calculate final averages.
# Tone is all from our method
Tone = [0.05159, 0.02054,0.00451,  0.09918]
# Even indexes are from our method, odd indexes are from the nvdiffrecmc first estimation
PSNR = [14.13055, 11.57039, 17.70357, 11.88978, 23.31084, 17.43790, 23.83234, 14.70438]
SSIM = [0.91713, 0.89379, 0.88372, 0.83522, 0.93701, 0.91191, 0.96024, 0.93736]
LPIPS = [0.35030, 0.07537, 0.19866, 0.13175, 0.13735, 0.09938, 0.02103, 0.03444]


# calculate and print average of elements in Tone list
print(f"\nOur avg tone Loss: {sum(Tone)/len(Tone)}")
print_average(PSNR, "PSNR")
print_average(SSIM, "SSIM")
print_average(LPIPS, "LPIPS")


############################################################################################################
# Plotting memory usage for whole pipeline, Nvdiffrecmc + OLAT + JOINT
############################################################################################################
# loading data
scene_name = "nerfactor_ficus"
out_dir = root_dir + f"/out/{scene_name}"
with open(out_dir + '/_JOINT/checkpoint/metrics_2022-12-31_17-47.pkl', 'rb') as f:
    ficus_joint = pickle.load(f)
with open(out_dir + '/_OLAT/checkpoint/metrics_2022-12-31_16-56.pkl', 'rb') as f:
    ficus_olat = pickle.load(f)
with open(out_dir + "/metrics_2023-01-02_16-23_dmtet_pass1.pkl", 'rb') as f:
    ficus_phase1nv = pickle.load(f)
with open(out_dir + "/metrics_2023-01-02_16-54_mesh_pass.pkl", 'rb') as f:
    ficus_phase2nv = pickle.load(f)

ficus_joint = keep_only_memory_log(ficus_joint)
ficus_olat = keep_only_memory_log(ficus_olat)
ficus_phase1nv = keep_only_memory_log(ficus_phase1nv)
ficus_phase2nv = keep_only_memory_log(ficus_phase2nv)
# putting together the two phases of nvdiffrecmc
ficus_phase1nv['max_all_mem'] = ficus_phase1nv['max_all_mem'] + ficus_phase2nv['max_all_mem']
ficus_phase1nv['init_mem'] = ficus_phase1nv['init_mem'] + ficus_phase2nv['init_mem']
ficus_phase1nv['post_backward_mem'] = ficus_phase1nv['post_backward_mem'] + ficus_phase2nv['post_backward_mem']
ficus_phase1nv['pre_render_mem'] = ficus_phase1nv['pre_render_mem'] + ficus_phase2nv['pre_render_mem']
ficus_nv = ficus_phase1nv
# removing from each dictionary the key "init_mem" because they are not needed
ficus_joint.pop('init_mem')
ficus_olat.pop('init_mem')
ficus_nv.pop('init_mem')

plot_pipeline_mem_analysis(ficus_nv, ficus_olat, ficus_joint, "linear")






