'''
analyze the interested metrics by the rings area in the image
'''
import sys
sys.path.append('../shortcut_skinseg/')

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image

import seaborn as sns
sns.set_theme(style='whitegrid')


from global_config import *
from train.loss_func import dice_loss, DiceLoss, CombinedLoss


def get_metrics(metrics_name):
    if metrics_name == 'dice':
        return DiceLoss()
    else:
        raise NotImplementedError


def get_mask(img_path,device):
    '''
    read png file and return the tensor
    '''
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    nparray = np.array(img)
    binary_mask = (nparray > 0).astype(np.uint8) 
    tensor_ = torch.tensor(binary_mask,device = device)
    tensor_ = tensor_.unsqueeze(0)
    # print(f'{tensor_.shape=}')
    # print(f'{tensor_}')
    return tensor_


def get_ring_area(mask,each_dis,size,interval):
    '''
    get the new mask with the same size of mask, while only keeping the ring areas between [each_distance, each_distance-interval]
    '''
    out_edge = each_dis
    out_gap = size//2 - out_edge
    inner_edge = max(each_dis - interval,0)
    inner_gap = size//2 - inner_edge
    new_mask = torch.zeros_like(mask)
    # print(f'{out_edge=}')
    new_mask[:,out_gap:size-out_gap,out_gap:size-out_gap] = mask[:,out_gap:size-out_gap,out_gap:size-out_gap]
    new_mask[:,inner_gap:size-inner_gap,inner_gap:size-inner_gap] = torch.zeros_like(mask[:,inner_gap:size-inner_gap,inner_gap:size-inner_gap])
    return new_mask


def compute_metrics_by_ring(gt_mask,pred_mask,func_metrics, interval=10,idx=None,):
    metrics_res = func_metrics(gt_mask,pred_mask)
    size = gt_mask.shape[-1]
    distance_to_centre_list = np.arange(size//2,0,-interval)
    distance_to_centre_list.sort()
    interested_metrics_list = []


    for each_dis in distance_to_centre_list:
        new_gt_mask = get_ring_area(gt_mask,each_dis,size,interval)
        new_pred_mask = get_ring_area(pred_mask,each_dis,size,interval)

        

        metrics_res = func_metrics(new_gt_mask,new_pred_mask)
        interested_metrics_list.append(metrics_res.cpu().detach().numpy())
        

    return distance_to_centre_list,interested_metrics_list



def plot(distance_to_centre_list,interested_metrics_list_all,save_name):
    plt.figure(figsize=(4,3),dpi=300)


    interested_metrics_np_all = np.array([np.array(xi) for xi in interested_metrics_list_all])
    interested_metrics_np_all = 1.0-interested_metrics_np_all
    avg_ = np.mean(interested_metrics_np_all,axis=0)
    # print(f'{avg_.shape=}')
    df = pd.DataFrame(interested_metrics_np_all) # Transpose to get groups as columns
    df.columns = distance_to_centre_list  # Set column names as labels

    # Melting the DataFrame to long format, which works well with sns.lineplot
    df_long = df.melt(var_name='Group', value_name='Value')

    # Plotting
    sns.lineplot(data=df_long, x='Group', y='Value', estimator=np.mean, ci='sd', marker='o')
    # plt.plot(distance_to_centre_list,avg_,color='red',marker='o')
    plt.ylabel('Dice Score')
    plt.xlabel('Distance from center')
    plt.ylim(0.0,1.0)
    plt.tight_layout()
    plt.savefig(REPO_PATH+'/records/analysis/metrics_by_rings/'+ save_name+'_lineplot.pdf')

    plt.figure(figsize=(4,3),dpi=300)
    df_4sns = pd.DataFrame(interested_metrics_np_all)  # Transpose to get groups as columns
    df_4sns.columns = distance_to_centre_list
    sns.boxplot(df_4sns,
                notch=True, showcaps=False,
                flierprops={"marker": "x"},
                boxprops={"facecolor": (.3, .5, .7, .5)},
                medianprops={"color": "r", "linewidth": 2},
                fliersize=0,
                )  

    plt.ylabel('Dice Score')
    plt.xlabel('Distance from center')
    plt.tight_layout()
    plt.savefig(REPO_PATH+'/records/analysis/metrics_by_rings/'+ save_name+'_boxplot.pdf')
        

def main():
    print('Start the analysis of the interested metrics by the rings area in the image')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    if not os.path.exists(REPO_PATH+'/records/analysis/metrics_by_rings/'):
        os.makedirs(REPO_PATH+'/records/analysis/metrics_by_rings/')

    # get all the pred and gt mask based on the version choosen
    version_num = 10 #TODO: change the version to the one you want to test on
    from_folder = 'inference' #['trained_models','inference']
    from_dataset_type = 'ISIC2017' #['ISIC2016','ISIC2017']
    metrics = 'dice' # ['dice','combined']
    interval = 10

    func_metrics= get_metrics(metrics)

    version_folder = REPO_PATH+f'/records/{from_folder}/{from_dataset_type}/unet/version_{version_num}/'

    gt_folder = version_folder + 'gt_mask'
    pred_folder = version_folder + 'pred_mask'

    sample_ids = [each.split('_')[1] for each in os.listdir(gt_folder)]
    sample_ids.sort()
    interested_metrics_list_all = []

    for idx,each_sample_id in enumerate(tqdm(sample_ids)):
        gt_mask = get_mask(os.path.join(gt_folder,'ISIC_'+each_sample_id+'_gt_mask.png'),device=device).float()
        pred_mask = get_mask(os.path.join(pred_folder,'ISIC_'+each_sample_id+'_pred_mask.png'),device=device).float()

        distance_to_centre_list, interested_metrics_list = compute_metrics_by_ring(gt_mask,pred_mask,func_metrics,idx=idx,interval=interval)
        interested_metrics_list_all.append(interested_metrics_list)


    plot(distance_to_centre_list,interested_metrics_list_all,f'{from_dataset_type}_{from_folder}_ver{version_num}_{metrics}_inter{interval}')

    return



if __name__ == '__main__':
    main()
