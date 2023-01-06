############################################################################################################
# EXPERIMENTS ANALYSIS UTILS FUNCTIONS
############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import os

def print_average(list, name):
    """
    function that given a list of float it prints the average of its elements
    with odd index and the average of its elements with even index
    @param list: list of float to calculate the average
    @param name: name of the list
    @return: None
    """
    odd = []
    even = []
    for i, element in enumerate(list):
        if i % 2 == 0:
            even.append(element)
        else:
            odd.append(element)
    print(f"Avg {name} nvdiffrecmc: " + str(sum(odd)/len(odd)))
    print(f"Avg {name} ours: " + str(sum(even)/len(even)))


def keep_only_metrics_no_memory_log(metrics):
    """
    function that takes as input a dictionary and returns a dictionary containing only the metrics
    @param metrics: dictionary containing the metrics and the memory log
    @return: dictionary containing only the metrics
    """
    return {key: value for key, value in metrics.items() if "mem" not in key}

def keep_only_memory_log(metrics):
    """
    function that takes as input a dictionary and returns a dictionary containing only the memory log
    @param metrics:  dictionary containing the metrics and the memory log
    @return: dictionary containing only the memory log
    """
    return {key: value for key, value in metrics.items() if "mem" in key}

def ewma_list(x, span=100):
    """
    function that takes as input a list of float and return the exponential weighted moving average of the list
    @param x: list of float containing the data
    @param span: window size for the exponential weighted moving average
    @return: list of float containing the exponential weighted moving average of the input list
    """
    x = np.array(x)
    weights = np.exp(np.linspace(-1., 0., span))
    weights /= weights.sum()
    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:span] = a[span]
    return a

def ewma_np(data, window=5):
    """
    function that takes as input a numpy array and returns the exponential weighted moving average of the array
    @param data: numpy array containing the data
    @param window:  window size for the exponential weighted moving average
    @return:  numpy array containing the exponential weighted moving average of the input array
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


############################################################################################################
# DIFFERENT PLOTTING FUNCTIONS FOR EACH EXPERIMENT
############################################################################################################

def plot_rgb_batch_exp(data1: dict, data2: dict, scale: str):
    keys = data1.keys()
    keys = [key for key in keys if key != "iter_time"]
    fig, axs = plt.subplots(int(len(keys)/2), 2)
    for i, key in enumerate(keys):
        axs.flat[i].plot(ewma_np(np.array(data1[key]),30), label="JOINT 250")
        axs.flat[i].plot(ewma_np(np.array(data2[key]),30), label="JOINT 500")
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
        axs.flat[i].set_title(key)

    fig.text(0.5, 0.04, 'Iterations', ha='center')
    fig.text(0.04, 0.5, 'Metrics', va='center', rotation='vertical')

    fig = plt.gcf()
    fig.set_size_inches(4*len(keys), 3*len(keys))
    plt.show()

def plot_OLAT_num_train_images_exp(data1: dict, data2: dict, data3: dict, scale: str):
    x_values = list(range(0, min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))*250, 250))
    x_values = [float(x)/1000 for x in x_values]
    data1['psnr_val'] = data1['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data2['psnr_val'] = data2['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data3['psnr_val'] = data3['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data1['ssim_val'] = data1['ssim_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data2['ssim_val'] = data2['ssim_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data3['ssim_val'] = data3['ssim_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data1['lpips_val'] = data1['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']), len(data3['lpips_val']))]
    data2['lpips_val'] = data2['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']), len(data3['lpips_val']))]
    data3['lpips_val'] = data3['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']), len(data3['lpips_val']))]



    keys = data1.keys()
    keys_to_keep = [key for key in keys if key in ["psnr_val", "ssim_val", "lpips_val"]]
    data1 = {key: value for key, value in data1.items() if key in keys_to_keep}
    data2 = {key: value for key, value in data2.items() if key in keys_to_keep}
    data3 = {key: value for key, value in data3.items() if key in keys_to_keep}

    plt.rcParams.update({'font.size': 15})
    y_label = ["PSNR(dB)", "SSIM", "LPIPS"]
    #a list with three colours easily distinguishable from each other, also for color-blind people
    colours = ["#000000", "#FF0000", "#00FF00"]
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Comparison OLAT step with different number of training images")
    fig.subplots_adjust(hspace=0.6)

    fig.subplots_adjust(wspace=0.2)


    for i, key in enumerate(data1.keys()):
        axs.flat[i].plot(x_values, ewma_np(np.array(data1[key])), label= "1650 images", color=colours[0])
        axs.flat[i].plot(x_values, ewma_np(np.array(data2[key])), label= "1100 images", color=colours[1])
        axs.flat[i].plot(x_values, ewma_np(np.array(data3[key])), label= "2100 images", color=colours[2])
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
        axs.flat[i].set_ylabel(y_label[i])

    plt.rcParams.update({'font.size': 16})

    fig.text(0.5, 0.01, 'Iterations (x1000)', ha='center')
    fig = plt.gcf()
    fig.set_size_inches(4*len(keys), 5)
    plt.subplots_adjust(bottom=0.15, left=0.03, right=0.97, top=0.85)

    plt.show()
    cwd = os.getcwd()
    fig.savefig(cwd + "/figures/comparison_olat_step.png", dpi=300)


def plot_OLAT_batch_size_exp(data1: dict, data2: dict, data3: dict, scale: str):
    x_values = list(range(0, min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))*250, 250))
    x_values = [float(x)/1000 for x in x_values]
    data1['psnr_val'] = data1['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data2['psnr_val'] = data2['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data3['psnr_val'] = data3['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data1['ssim_val'] = data1['ssim_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data2['ssim_val'] = data2['ssim_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data3['ssim_val'] = data3['ssim_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']), len(data3['psnr_val']))]
    data1['lpips_val'] = data1['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']), len(data3['lpips_val']))]
    data2['lpips_val'] = data2['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']), len(data3['lpips_val']))]
    data3['lpips_val'] = data3['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']), len(data3['lpips_val']))]



    keys = data1.keys()
    keys_to_keep = [key for key in keys if key in ["psnr_val", "ssim_val", "lpips_val"]]
    data1 = {key: value for key, value in data1.items() if key in keys_to_keep}
    data2 = {key: value for key, value in data2.items() if key in keys_to_keep}
    data3 = {key: value for key, value in data3.items() if key in keys_to_keep}

    plt.rcParams.update({'font.size': 15})
    y_label = ["PSNR(dB)", "SSIM", "LPIPS"]
    # create a list with three colours easily distinguishable from each other, also for color-blind people
    colours = ["#000000", "#FF0000", "#00FF00"]
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Comparison OLAT step with different batch sizes")
    fig.subplots_adjust(hspace=0.6)
    fig.subplots_adjust(wspace=0.2)


    for i, key in enumerate(data1.keys()):
        axs.flat[i].plot(x_values, ewma_np(np.array(data1[key])), label= "Batch=500", color=colours[0])
        axs.flat[i].plot(x_values, ewma_np(np.array(data2[key])), label= "Batch=10k", color=colours[1])
        axs.flat[i].plot(x_values, ewma_np(np.array(data3[key])), label= "Full image", color=colours[2])
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
        axs.flat[i].set_ylabel(y_label[i])

    plt.rcParams.update({'font.size': 16})

    fig.text(0.5, 0.01, 'Iterations (x1000)', ha='center')

    fig = plt.gcf()
    fig.set_size_inches(4*len(keys), 5)
    plt.subplots_adjust(bottom=0.15, left=0.03, right=0.97, top=0.85)

    plt.show()
    cwd = os.getcwd()
    fig.savefig(cwd + "/figures/comparison_olat_batch.png", dpi=300)


def plot_envmap_lr_exp(data1: dict, data2: dict, data3: dict, scale: str):
    x_values = list(range(0, min(len(data1['psnr_val']), len(data2['psnr_val']))*100, 100))
    x_values = [float(x)/1000 for x in x_values]
    data1['loss_olat'] = data1['loss_olat'][:min(len(data1['loss_olat']), len(data2['loss_olat']))]
    data1['loss_env'] = data1['loss_env'][:min(len(data1['loss_env']), len(data2['loss_env']))]
    data2['loss_olat'] = data2['loss_olat'][:min(len(data1['loss_olat']), len(data2['loss_olat']))]
    data2['loss_env'] = data2['loss_env'][:min(len(data1['loss_env']), len(data2['loss_env']))]
    data3['loss_olat'] = data3['loss_olat'][:min(len(data1['loss_olat']), len(data2['loss_olat']))]
    data3['loss_env'] = data3['loss_env'][:min(len(data1['loss_env']), len(data2['loss_env']))]

    keys = data1.keys()
    keys_to_keep = [key for key in keys if key in ["loss_olat", "loss_env"]]
    data1 = {key: value for key, value in data1.items() if key in keys_to_keep}
    data2 = {key: value for key, value in data2.items() if key in keys_to_keep}
    data3 = {key: value for key, value in data3.items() if key in keys_to_keep}
    plt.rcParams.update({'font.size': 15})
    y_label = ["OLAT Loss", "Env Loss"]
    # create a list with three colours easily distinguishable from each other, also for color-blind people
    colours = ["#000000", "#FF0000", "#00FF00"]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Comparison JOINT step with different envmap learning rates")
    fig.subplots_adjust(hspace=0.6)

    fig.subplots_adjust(wspace=0.2)

    for i, key in enumerate(data1.keys()):
        axs.flat[i].plot(x_values, ewma_np(np.array(data1[key]),window=10), label= "lr=1", color=colours[0])
        axs.flat[i].plot(x_values, ewma_np(np.array(data2[key]),window=10), label= "lr=0.2", color=colours[1])
        axs.flat[i].plot(x_values, ewma_np(np.array(data3[key]),window=10), label= "lr=0.02", color=colours[2])
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
        axs.flat[i].set_ylabel(y_label[i])

    plt.rcParams.update({'font.size': 16})

    fig.text(0.5, 0.01, 'Iterations (x1000)', ha='center')

    fig = plt.gcf()
    fig.set_size_inches(2*len(keys), 5)
    plt.subplots_adjust(bottom=0.15, left=0.08, right=0.92, top=0.85)

    plt.show()
    cwd = os.getcwd()
    fig.savefig(cwd + "/figures/comparison_joint_envmapLR.png", dpi=300)

def plot_100kOLAT_50kJOINT_exp(data1: dict, data2: dict,  scale: str):
    x_values_1 = list(range(0, len(data1['psnr_val']) * 250, 250))
    x_values_1 = [float(x)/1000 for x in x_values_1]
    x_values_2 = list(range(0, min(len(data1['psnr_val']), len(data2['psnr_val'])) * 100, 100))
    x_values_2 = [float(x) / 1000 for x in x_values_2]
    data2['loss_train'] = data2.pop('loss')
    data2['loss_train'] = data2['loss_train'][:min(len(data1['loss_train']), len(data2['loss_train']))]
    data2['psnr_val'] = data2['psnr_val'][:min(len(data1['psnr_val']), len(data2['psnr_val']))]
    data2['ssim_val'] = data2['ssim_val'][:min(len(data1['ssim_val']), len(data2['ssim_val']))]
    data2['lpips_val'] = data2['lpips_val'][:min(len(data1['lpips_val']), len(data2['lpips_val']))]

    keys = data1.keys()
    keys_to_keep = [key for key in keys if key in ["loss_train", "psnr_val", "ssim_val", "lpips_val"]]
    data1 = {key: value for key, value in data1.items() if key in keys_to_keep}
    data2 = {key: value for key, value in data2.items() if key in keys_to_keep}

    plt.rcParams.update({'font.size': 15})
    y_label = ["Train Loss", "PSNR", "SSIM", "LPIPS"]
    # list with three colours easily distinguishable for color-blind people
    colours = ["#000000", "#FF0000", "#00FF00"]
    fig, axs = plt.subplots(1, 4)
    fig.suptitle("Convergence analysis of JOINT and OLAT training")
    fig.subplots_adjust(hspace=0.6)
    fig.subplots_adjust(wspace=0.3)

    for i, key in enumerate(data1.keys()):
        axs.flat[i].plot(x_values_1, ewma_np(np.array(data1[key]), window=30), label="OLAT step", color=colours[0])
        axs.flat[i].plot(x_values_2, ewma_np(np.array(data2[key]), window=30), label="JOINT step", color=colours[1])
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
        axs.flat[i].set_ylabel(y_label[i])


    plt.rcParams.update({'font.size': 16})

    fig.text(0.5, 0.01, 'Iterations (x1000)', ha='center')
    fig = plt.gcf()
    fig.set_size_inches(4*len(keys), 5)
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.97, top=0.85)

    plt.show()
    cwd = os.getcwd()
    fig.savefig(cwd + "/figures/olat_joint_150k.png", dpi=300)


def plot_noOLAT_onlyJOINT_exp(data1: dict, scale: str):
    keys = data1.keys()
    keys = [key for key in keys if key != "iter_time"]
    fig, axs = plt.subplots(int(len(keys)/2), 2)
    for i, key in enumerate(keys):
        axs.flat[i].plot(ewma_np(np.array(data1[key])), label="hotdog_olat 100k")
        axs.flat[i].set_yscale(scale)
        axs.flat[i].legend()
        axs.flat[i].set_title(key)


    fig.text(0.5, 0.04, 'Iterations', ha='center')
    fig.text(0.04, 0.5, 'Metrics', va='center', rotation='vertical')

    fig = plt.gcf()
    fig.set_size_inches(4*len(keys), 3*len(keys))
    plt.show()


def plot_pipeline_mem_analysis(data1: dict, data2: dict,  data3: dict, scale: str):

    plt.rcParams.update({'font.size': 15})
    y_label = ["Nvdiffrecmc", "OLAT Training", "JOINT Training"]
    colours = ["#000000", "#FF0000", "#00FF00", "#0000FF"]
    labels = ["pre-render", "post-backward", "max allocated"]
    minimums = [0.5, 1, 0.8]
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Analysis of memory consumption across all training steps")
    fig.subplots_adjust(hspace=0.6)

    fig.subplots_adjust(wspace=0.2)

    for i, data in enumerate([data1, data2, data3]):
        for j, (key, value) in enumerate(data.items()):
            axs[i].set_ylim(minimums[i], np.array(data["max_all_mem"]).max()+0.1)
            axs[i].plot(ewma_list(value), label=labels[j], color=colours[j])
            axs[i].fill_between(range(len(value)), ewma_list(value), color=colours[j], alpha=0.1*(4-j))
        axs[i].set_title(y_label[i])
        axs[i].set_yscale(scale)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.92), fancybox=True, shadow=True,
               prop={'size': 15})
    plt.rcParams.update({'font.size': 16})

    fig.text(0.5, 0.02, 'Iterations', ha='center')
    fig.text(0.01, 0.5, 'Memory usage (GB)', va='center', rotation='vertical')
    fig = plt.gcf()
    fig.set_size_inches(8*3, 5)
    plt.subplots_adjust(bottom=0.15, left=0.04, right=0.96, top=0.75)

    plt.show()
    cwd = os.getcwd()
    fig.savefig(cwd + "/figures/memory_analysis.png", dpi=300)


