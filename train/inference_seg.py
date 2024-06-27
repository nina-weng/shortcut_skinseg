'''
inference with trained model

'''
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader


import sys
sys.path.append('../shortcut_skinseg/')
from global_config import *
from train.utils import *
from inference_config import inferSegConfig
from train.dataloader import SkinSegDataset
from models.UNet import UNet
from train.train_seg import test_func,get_loss_func

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'inference segmentation task on {device}')

    # records and configs
    model_name = 'unet'
    record_path = REPO_PATH+f'/records/inference/{inferSegConfig.dataset_type}/{model_name}/'
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    version_num_str = get_version_num(record_path)
    record_path = os.path.join(record_path,'version_'+version_num_str)
    os.makedirs(record_path)

    print('inference results store in: {}'.format(record_path))

    save_config(inferSegConfig,os.path.join(record_path,'config_record.txt'))
    print('save config.')

    # test dataloader
    df_test_extend = pd.read_csv(REPO_PATH+f'/datafiles/{inferSegConfig.dataset_type}_test_seg.csv',dtype=str)


    # Create datasets
    test_dataset = SkinSegDataset(dataset_dir=DATASET_PATH,
                                  isCrop=inferSegConfig.isCrop,
                                data_df=df_test_extend, 
                                resize_to = inferSegConfig.resized_size,
                                random_crop=False)


    print(f'#test:{len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=inferSegConfig.batch_size, shuffle=False)

    # load the trained model
    model_eva = UNet(n_channels=3, n_classes=2)  # Initialize your U-Net model here
    model_eva.to(device)
    model_eva.load_state_dict(torch.load(REPO_PATH+inferSegConfig.load_model_path))
    model_eva.eval()

    
    # inference
    loss_func = get_loss_func(inferSegConfig.loss_type)
    test_func(model=model_eva, 
              loss_type = inferSegConfig.loss_type,
              loss_func= loss_func,
              data_loader=test_loader, 
              device=device,
              save_dir=record_path,
              save_gt_mask=inferSegConfig.save_gt_mask)


    print('Finished job')

    return





if __name__ == '__main__':
    main()