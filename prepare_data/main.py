'''
calculate the centre of the annotation of every segementation
output a csv file [name, centre of the segmentation]
'''
import numpy as np
import os
import imageio
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme(style='white')

import sys
sys.path.append('../shortcut_skinseg/')
from global_config import *



def centre_one_seg(mask,reshape_to=(256,256)):
    '''
    get the centre (x,y) of the seg
    '''
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)


    rows, cols = np.where(mask == 255)

    # Compute the average of the row indices and the column indices to find the center
    center_y = np.mean(rows)
    center_x = np.mean(cols)


    return (center_y, center_x)


# def get_centres(masks):
#     print('Compute the centre.')
#     cs = []
#     for m in masks:
#         c = centre_one_seg(m)
#         print(c)
#         cs.append(c)
#     return cs


def load_segs(data_folder):
    # print('Loading seg masks.')
    image_names = os.listdir(data_folder)
    # image_names = image_names[:10]
    centres = []
    imag_sizes = []
    for iname in tqdm(image_names):
        path = data_folder + '/' + iname
        mask = Image.open(path)
        # Resize the image to the new dimensions
        # resized_mask = mask.resize(reshape_to)

        # Convert the PIL image to a NumPy array
        mask_array = np.array(mask)

        c = centre_one_seg(mask_array)
        # print(c)
        centres.append(c)
        imag_sizes.append(mask_array.shape)

    return image_names,imag_sizes,centres


def write_csv(list_image_names, imag_sizes,list_centres, save_path):
    combined_data = [(s, *t, *p) for s, t, p in zip(list_image_names, imag_sizes,list_centres)]
    # print(combined_data)

    # Create the DataFrame
    df = pd.DataFrame(combined_data, columns=['image_names','img_size_x','img_size_y','centre_x','centre_y'])
    df.to_csv(save_path,index=False)
    return 


def get_img_path(dataset_type,split,image_type,img_id):
    if dataset_type == 'ISIC2016':
        if split == 'train':
            if image_type == 'image':
                return 'ISBI2016_ISIC_Part1_Training_Data/'+'ISIC_' + img_id + '.jpg'
            elif image_type == 'mask':
                return 'ISBI2016_ISIC_Part1_Training_GroundTruth/' + 'ISIC_' + img_id + '_Segmentation.png'
        elif split == 'test':
            if image_type == 'image':
                return 'ISBI2016_ISIC_Part1_Test_Data/'+'ISIC_' + img_id + '.jpg'
            elif image_type == 'mask':
                return 'ISBI2016_ISIC_Part1_Test_GroundTruth/'+'ISIC_' + img_id + '_Segmentation.png'
    elif dataset_type == 'ISIC2017':
        if split == 'train':
            if image_type == 'image':
                return 'ISIC-2017_Training_Data/'+'ISIC_' + img_id + '.jpg'
            elif image_type == 'mask':
                return 'ISIC-2017_Training_Part1_GroundTruth/' + 'ISIC_' + img_id + '_segmentation.png'
        elif split == 'test':
            if image_type == 'image':
                return 'ISIC-2017_Test_v2_Data/'+'ISIC_' + img_id + '.jpg'
            elif image_type == 'mask':
                return 'ISIC-2017_Test_v2_Part1_GroundTruth/'+'ISIC_' + img_id + '_segmentation.png'
        elif split == 'validation':
            if image_type == 'image':
                return 'ISIC-2017_Validation_Data/'+'ISIC_' + img_id + '.jpg'
            elif image_type == 'mask':
                return 'ISIC-2017_Validation_Part1_GroundTruth/'+'ISIC_' + img_id + '_segmentation.png'
    else:
        raise NotImplemented
    
    


def add_img_path(df,dataset_type,split,):
    
    df['image_id'] = df['image_names'].apply(lambda x: x.split('_')[1])

    df['image_path'] = df['image_id'].apply(lambda x: f'{dataset_type}/'+get_img_path(dataset_type=dataset_type, split=split,image_type='image',img_id=x))
    df['mask_path'] = df['image_id'].apply(lambda x: f'{dataset_type}/'+get_img_path(dataset_type=dataset_type, split=split,image_type='mask',img_id=x))

    return df


def main():
    dataset_type = 'ISIC2016'
    resize_to = 256 # define resized image size
    print(f'Processing dataset : {dataset_type} with {resize_to=}. Computing the centre of the image and store the data as csv file under \'./datafiles\'.')
    print('*'*30)
    print('Part 1: calculate annotation centre. ')

    if dataset_type == 'ISIC2016':
        # training part
        splits_set = ['train','test']
        train_seg_folder = DATASET_PATH + f'/{dataset_type}/ISBI2016_ISIC_Part1_Training_GroundTruth/'
        test_seg_folder = DATASET_PATH + f'/{dataset_type}/ISBI2016_ISIC_Part1_Test_GroundTruth/'
        
    elif dataset_type == 'ISIC2017':
        splits_set = ['train','validation','test']
        train_seg_folder = DATASET_PATH + f'/{dataset_type}/ISIC-2017_Training_Part1_GroundTruth/'
        test_seg_folder = DATASET_PATH + f'/{dataset_type}/ISIC-2017_Test_v2_Part1_GroundTruth/'
        val_seg_folder = DATASET_PATH + f'/{dataset_type}/ISIC-2017_Validation_Part1_GroundTruth/'
    else: 
        raise NotImplemented

    for split in splits_set:
        print(f'---{split=}---')
        if split == 'train':
            img_names,image_sizes,centres = load_segs(train_seg_folder)
            write_csv(img_names,image_sizes,centres,save_path=REPO_PATH+f'/datafiles/{dataset_type}_{split}_seg.csv')
        elif split == 'test':
            img_names,image_sizes,centres = load_segs(test_seg_folder)
            write_csv(img_names,image_sizes,centres,save_path=REPO_PATH+f'/datafiles/{dataset_type}_{split}_seg.csv')
        elif split == 'validation':
            img_names,image_sizes,centres = load_segs(val_seg_folder)
            write_csv(img_names,image_sizes,centres,save_path=REPO_PATH+f'/datafiles/{dataset_type}_{split}_seg.csv')

    print('Finish Part 1: calculate annotation centre. ')
    print('*'*30)
    print('Part 2: add image path and add the mask postion label')

    
    for split in splits_set:
        print(f'---{split=}---')
        df = pd.read_csv(REPO_PATH+f'/datafiles/{dataset_type}_{split}_seg.csv')
        # add the image path
        df = add_img_path(df,dataset_type,split)
        # add the mask position label
        def resize(from_size,to,x):
            return x*to/from_size

        df['centre_x_resize'] = df.apply(lambda x: resize(from_size=x['img_size_x'],to=resize_to,x=x['centre_x']), axis=1)
        df['centre_y_resize'] = df.apply(lambda x: resize(from_size=x['img_size_y'],to=resize_to,x=x['centre_y']), axis=1)

        def get_distance(x1,y1,x2,y2):
            return math.hypot(x2 - x1, y2 - y1)

        df['distance'] = df.apply(lambda x: get_distance(x['centre_x_resize'],x['centre_y_resize'],resize_to/2,resize_to/2), axis=1)

        def assign_mask_pos(d):
            if d<=20:
                return 'centre'
            elif d<40:
                return 'middle'
            else:
                return 'off-centre'


        df['mask_position'] = df['distance'].apply(lambda x:assign_mask_pos(x))

        df =df.sort_values(by='image_names')
        df = df[['image_names', 'img_size_x', 'img_size_y', 'centre_x',
            'centre_y', 'centre_x_resize', 'centre_y_resize', 'distance',
            'mask_position','image_id','image_path','mask_path']]
        
        df.to_csv(REPO_PATH+f'/datafiles/{dataset_type}_{split}_seg.csv',index=False)

        # plot the hist
        plt.figure(figsize=(5,5),dpi=100)
        df['distance'].hist()
        plt.xlabel(f'centre of segs to the centre of the img in pixel\n(resize to {resize_to})')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(REPO_PATH+f'/datafiles/datastat/distance_hist_{dataset_type}_{split}.png')

        plt.figure(figsize=(5,5),dpi=100)
        df['mask_position'].hist()
        plt.xlabel(f'mask label')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(REPO_PATH+f'/datafiles/datastat/mask_pos_hist_{dataset_type}_{split}.png')


    
    print('Finish Part 2: add image path and add the mask postion label')
    print('*'*30)
    print('Part 3: prepare the extend test set. (including middle and off-centre samples from train and validation set, as we will not using which during training)')

    df_train = pd.read_csv(REPO_PATH+f'/datafiles/{dataset_type}_train_seg.csv', dtype = str)
    df_train_middle_offcentre = df_train[df_train['mask_position']!='centre']
    df_test = pd.read_csv(REPO_PATH+f'/datafiles/{dataset_type}_test_seg.csv',dtype = str)
    df_test_extend = pd.concat([df_test,df_train_middle_offcentre])

    if dataset_type == 'ISIC2017':
        df_val = pd.read_csv(REPO_PATH+f'/datafiles/{dataset_type}_validation_seg.csv', dtype = str)
        df_val_middle_offcentre = df_val[df_train['mask_position']!='centre']
        df_test_extend = pd.concat([df_test_extend,df_val_middle_offcentre])

    df_test_extend = df_test_extend.sort_values(by='image_names')
    df_test_extend = df_test_extend[['image_names', 'img_size_x', 'img_size_y', 'centre_x',
        'centre_y', 'centre_x_resize', 'centre_y_resize', 'distance',
        'mask_position','image_id','image_path','mask_path']]
    df_test_extend.to_csv(REPO_PATH+f'/datafiles/{dataset_type}_test_extend_seg.csv',index=False)
            
    
    print('Finish Part 3: prepare the extend test set.')






    return 


if __name__ == '__main__':
    main()