import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torchvision.utils import save_image

import sys
sys.path.append('../shortcut_skinseg/')
from global_config import *
from loss_func import dice_loss, DiceLoss, CombinedLoss
from train.dataloader import SkinSegDataset
from train.train_config import trainSegConfig
from models.UNet import UNet
from train.utils import *

def get_loss_func(str_):
    if str_ == 'dice':
        return DiceLoss()
    elif str_ == 'combined':
        return CombinedLoss()
    else: 
        raise NotImplemented



def test_func(model, 
              loss_type,
              loss_func,
              data_loader, 
              device,
              save_dir,
              save_gt_mask=False):
    '''
    inference, save the mask and record the loss.
    return: 
    - saved predicted mask in predicted_mask_save_dir
    - saved csv file in save_dir
    '''
    model.eval()

    predicted_mask_save_dir = os.path.join(save_dir,'pred_mask')
    os.makedirs(predicted_mask_save_dir)
    if save_gt_mask:
        gt_mask_save_dir = os.path.join(save_dir,'gt_mask')
        os.makedirs(gt_mask_save_dir)

    sample_ids=[]
    losses=[]
    mask_poses=[]

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, mask = batch['image'].to(device), batch['mask'].to(device)
            sample_id,mask_pos = batch['sample_id'],batch['mask_pos']

            out = model(img)

            # save the predicted mask
            for idx in range(len(sample_id)):
                id_ = sample_id[idx]
                out_ = out[idx]
                # to class prob
                out_ = torch.sigmoid(out_)
                # to class
                out_ = torch.argmax(out_, dim=0).float()
                save_image(out_, predicted_mask_save_dir+f'/ISIC_{id_}_pred_mask.png')
                if save_gt_mask:
                    save_image(mask[idx], gt_mask_save_dir+f'/ISIC_{id_}_gt_mask.png')

            out = torch.sigmoid(out)
            loss = loss_func(out, mask, reduction = 'each') # we would like the list of the loss, (each sample)
            
            losses.append(loss)
            sample_ids.extend(sample_id)
            mask_poses.extend(mask_pos)

        losses = torch.cat(losses, dim=0)
        losses = torch.squeeze(losses)


    # write into csv
    df = pd.DataFrame({
            'ISIC_ID': sample_ids,
            loss_type: losses.cpu().detach().tolist(),
            'mask_pos': mask_poses
        })
    
    df.to_csv(os.path.join(save_dir,'prediction_loss_maskpos.csv'), index=False)

    return 


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'train segmentation task on {device}')

    model_name = 'unet'
    train_model_path = REPO_PATH+f'/records/trained_models/{trainSegConfig.dataset_type}/{model_name}/'
    if not os.path.exists(train_model_path):
        os.makedirs(train_model_path)
    version_num_str = get_version_num(train_model_path)
    train_model_path = os.path.join(train_model_path,'version_'+version_num_str)
    os.makedirs(train_model_path)

    print('train models and results store in: {}'.format(train_model_path))

    save_config(trainSegConfig,os.path.join(train_model_path,'config_record.txt'))
    print('save config.')


    df_train = pd.read_csv(REPO_PATH+f'/datafiles/{trainSegConfig.dataset_type}_train_seg.csv',dtype=str)
    df_test = pd.read_csv(REPO_PATH+f'/datafiles/{trainSegConfig.dataset_type}_test_seg.csv',dtype=str)
    if trainSegConfig.dataset_type == 'ISIC2017':
        df_val = pd.read_csv(REPO_PATH+f'/datafiles/{trainSegConfig.dataset_type}_validation_seg.csv',dtype=str)

    if trainSegConfig.training_filer == 'only_centre':
        df_train = df_train[df_train['mask_position']=='centre']
        # test set: should include all
        if trainSegConfig.dataset_type == 'ISIC2017':
            df_val = df_val[df_val['mask_position']=='centre']

    

    # Create datasets
    if trainSegConfig.dataset_type == 'ISIC2017':
        train_dataset = SkinSegDataset(dataset_dir=DATASET_PATH,
                                data_df=df_train, 
                                resize_to = trainSegConfig.resized_size,
                                isCrop=trainSegConfig.isCrop)
        val_dataset = SkinSegDataset(dataset_dir=DATASET_PATH,
                                data_df=df_val, 
                                resize_to = trainSegConfig.resized_size,
                                isCrop=trainSegConfig.isCrop)
    elif trainSegConfig.dataset_type == 'ISIC2016':
        train_val_dataset = SkinSegDataset(dataset_dir=DATASET_PATH,
                                    data_df=df_train, 
                                    resize_to = trainSegConfig.resized_size,
                                    isCrop=trainSegConfig.isCrop)
        # split the train and validation based on random seed
        generator = torch.Generator().manual_seed(trainSegConfig.random_seed)
        train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset,[0.8,0.2],generator)
    else: raise NotImplemented


    test_dataset = SkinSegDataset(dataset_dir=DATASET_PATH,
                                data_df=df_test, 
                                resize_to = trainSegConfig.resized_size,
                                isCrop=trainSegConfig.isCrop,
                                random_crop=False)

    
    

    print(f'#train:{len(train_dataset)}')
    print(f'#val:{len(val_dataset)}')
    print(f'#test:{len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=trainSegConfig.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainSegConfig.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=trainSegConfig.batch_size, shuffle=False)

    # Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=2)  # Initialize your U-Net model here
    model.to(device)

    # set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=trainSegConfig.lr)
    loss_func = get_loss_func(trainSegConfig.loss_type)


    num_epochs = trainSegConfig.num_epochs
    best_val_loss = None

    print('Starting to training...')
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader):
            images, masks = batch['image'], batch['mask']
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            loss = loss_func(outputs, masks)  # Replace or combine with Cross Entropy here
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                images, masks = batch['image'], batch['mask']
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                loss = loss_func(outputs, masks)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')
            if best_val_loss is None or val_loss < best_val_loss:
                # save the model if the val loss is smaller
                torch.save(model.state_dict(), train_model_path+'/unet_model.pth')

    
    # load the best performed model in validation set
    model_eva = UNet(n_channels=3, n_classes=2)  # Initialize your U-Net model here
    model_eva.to(device)
    model_eva.load_state_dict(torch.load(train_model_path+'/unet_model.pth'))
    model_eva.eval()
    

    # Test, save the predicted mask
    test_func(model=model_eva, 
              loss_type = trainSegConfig.loss_type,
              loss_func= loss_func,
              data_loader=test_loader, 
              device=device,
              save_dir=train_model_path,
              save_gt_mask=trainSegConfig.save_gt_mask)


    print('Finished job')
    return





if __name__ == '__main__':
    main()