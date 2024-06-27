'''
experiments on moving patch
'''

import os
import numpy as np
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as TF 

import sys
sys.path.append('../shortcut_skinseg/')

from global_config import *
from models.UNet import UNet

def resize_and_crop_image(image_path,resize_to=(256,256)):
    img = Image.open(image_path)
    resized_img = img.resize(resize_to)
    
    cropped_images = []

    for idx,each_s in enumerate(crop_starts):
        each_img = resized_img.crop((each_s, each_s, each_s + crop_size, each_s + crop_size))
        each_img = each_img.resize(resize_to)
        cropped_images.append(each_img)

    plt.show()
    return cropped_images


if __name__ == '__main__':
    # from ISIC2017 test set
    sample_img_list = ['ISIC_0014867.jpg',]
    data_dir = DATASET_PATH + 'ISIC2017/ISIC-2017_Test_v2_Data/'

    
    # save the sample images
    save_to = REPO_PATH +'records/analysis/moving_patch/samples/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for each_sample in sample_img_list:
        src = os.path.join(data_dir,each_sample)
        dst = os.path.join(save_to,each_sample)
        shutil.copy(src,dst)

    sample_img_path = save_to+sample_img_list[0]

    resize_to = (256,256)
    crop_size = 150
    crop_interval = 12
    crop_starts = np.arange(0,resize_to[0]-crop_size,crop_interval)


    cropped_images = resize_and_crop_image(sample_img_path)

    # load the segmentation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # records and configs
    dataset_type = 'ISIC2017'
    load_model_type = 'unet'
    load_model_version = 7 #TODO: change the version to the one you want to test on
    load_model_path = REPO_PATH+ f'/records/trained_models/{dataset_type}/{load_model_type}/version_{str(load_model_version)}/{load_model_type}_model.pth'


    # load the trained model
    model_eva = UNet(n_channels=3, n_classes=2)  # Initialize your U-Net model here
    model_eva.to(device)
    model_eva.load_state_dict(torch.load(load_model_path))
    model_eva.eval()

    

    compose_list = []
    resize_to = (256,256)

    compose_list.extend(
                    # every setting has the followings
                    [TF.ToTensor(),
                    TF.Resize((resize_to[0],resize_to[1]), 
                            interpolation=TF.InterpolationMode.BICUBIC, 
                            antialias=True),
                    TF.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1] 
                    ])
    transform = TF.Compose(compose_list)


    # get the predicted segmentation results
    preds = []
    for each_cropped_img in cropped_images:
        each_cropped_img = transform(each_cropped_img)
        each_cropped_img =each_cropped_img.to(device)
        # print(each_cropped_img.shape)
        out = model_eva(each_cropped_img.unsqueeze(0))[0]
        # to class prob
        out = torch.sigmoid(out)
        # to class
        out = torch.argmax(out, dim=0).float()
        preds.append(out.detach().cpu().numpy())
        
    

    # plot
    fig,axes = plt.subplots(2,len(preds),figsize=(4*len(preds),2*4))

    for idx,each_p in enumerate(preds):

        axes[0,idx].imshow(cropped_images[idx])
        axes[0,idx].axis('off')
        

        axes[1,idx].imshow(each_p)
        axes[1,idx].axis('off')

    sample_id = sample_img_path.split('/')[-1]
    plt.tight_layout(h_pad=3,w_pad=1)
    plt.savefig(REPO_PATH + '/records/analysis/moving_patch/{}_'.format(load_model_version)+sample_id)
    # plt.show()