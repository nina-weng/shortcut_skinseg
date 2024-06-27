
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as TF 
import torch
from torchvision.utils import save_image
import numpy as np


import sys
sys.path.append('../shortcut_skinseg/')
from global_config import *
from train.utils import *


class SkinSegDataset(Dataset):
    def __init__(self, dataset_dir,data_df, 
                 resize_to = (256,256),
                 isCrop = False,
                 random_crop = True):
        self.dataset_dir = dataset_dir
        self.data_df = data_df
        self.resize_to = resize_to
        self.isCrop = isCrop
        self.random_crop = random_crop
        self.img_transforms = self.get_transforms(data_type='image')
        self.mask_transforms = self.get_transforms(data_type='mask')

        self.img_file_names = []
        self.mask_file_names= []
        self.mask_positions=[]
        self.sample_ids = []

        for index,each in self.data_df.iterrows():
            image_path = each['image_path']
            mask_path = each['mask_path']
            img_id = each['image_id']
            self.img_file_names.append(os.path.join(self.dataset_dir,image_path))
            self.mask_file_names.append(os.path.join(self.dataset_dir,mask_path))
            self.mask_positions.append(each['mask_position'])
            self.sample_ids.append(img_id)


        
    def __len__(self):
        return len(self.img_file_names)
    
    def __getitem__(self, idx):
        img_path = self.img_file_names[idx]
        mask_path = self.mask_file_names[idx]
        sample_id = self.sample_ids[idx]
        mask_pos = self.mask_positions[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.isCrop:
            if self.random_crop:
                image,mask = self.crop(image,mask)
            else: # if not random crop
                image,mask = self.crop(image,mask,random_seed=idx)

        image = self.img_transforms(image) # Dimension (3, resizeto[0], resizeto[1])
        mask = self.mask_transforms(mask) # Dimension (1, resizeto[0], resizeto[1])

        


        return {'sample_id':sample_id, 'image':image, 'mask':mask, 'mask_pos':mask_pos}
    
    def crop(self,img,mask,random_seed=None):
        '''
        crop one image: randomly select one point in the masking area, and the randomly select one part of the four piles.
        '''
        point_in_mask = find_random_point_in_mask(mask,fix_middle_point=True)
        # print(f'{point_in_mask=}')
        cropped_image, cropped_mask = crop_image_at_point(img,mask, point_in_mask,random_seed)
        return cropped_image, cropped_mask 


    def get_transforms(self,data_type):
        compose_list = self.get_compose_list(data_type)
        transform = TF.Compose(compose_list)
        return transform
    
    def get_compose_list(self,data_type):
        '''
        get the compose function list from the parameters
        '''
        compose_list=[]

        if data_type == 'image':
            compose_list.extend(
                # every setting has the followings
                [TF.ToTensor(),
                TF.Resize((self.resize_to[0],self.resize_to[1]), 
                        interpolation=TF.InterpolationMode.BICUBIC, 
                        antialias=True),
                TF.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1] 
                ]
            )

        elif data_type == 'mask':
            compose_list.extend(
                # every setting has the followings
                [TF.ToTensor(),
                TF.Resize((self.resize_to[0],self.resize_to[1]), 
                        interpolation=TF.InterpolationMode.NEAREST, 
                        antialias=True),
                ]
            )

        return compose_list
    
    def exam(self, idx_list: list[int],
                save_dir: str)-> None:
        '''
        exam (the augmentaion) by saving some images from train set
        '''
        for idx in idx_list:
            one_sample = self. __getitem__(idx=idx)
            img = one_sample['image']
            mask = one_sample['mask']
            save_image(img, save_dir+f'/idx{idx}_img.png')
            save_image(mask, save_dir+f'/idx{idx}_mask.png')
            img = inverse_transform(img)
            img = img/255.0
            save_image(img, save_dir+f'/idx{idx}_img_inv.png')
        return
    


if __name__ == '__main__':
    print('test dataloader')

    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(REPO_PATH+'/datafiles/ISIC2017_test_seg.csv',dtype=str)
    skinseg_test = SkinSegDataset(dataset_dir=DATASET_PATH,
                                data_df=df, 
                                resize_to = (256,256),
                                isCrop=False,
                                )
    
    train_dataloader = DataLoader(skinseg_test, batch_size=8, shuffle=True)

    for i_step, (batch) in enumerate(tqdm(train_dataloader)):
        img = batch['image']
        mask = batch['mask']
        print(f'{img.shape=},{mask.shape=}')
        img_0 = img[0]
        mask_0 = mask[0]
        print(f'{torch.max(img_0)=},{torch.min(img_0)=}')
        print(f'{torch.max(mask_0)=},{torch.min(mask_0)=}')


        break

    save_dir = REPO_PATH+'/records/dataloader_test_noncrop/'
    os.makedirs(save_dir)
    idx_list = range(0,5,1)
    skinseg_test.exam(idx_list,save_dir=save_dir)
