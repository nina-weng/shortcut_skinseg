from dataclasses import dataclass


@dataclass
class TrainSegConfig:
    dataset_type: str = '' #['ISIC2016', 'ISIC2017']
    # resize to
    resized_size: tuple[int] = (256,256)
    # training filter 
    training_filer: str = 'all' # ['all','only_centre']
    # 
    random_seed:int =2024
    # training detaills
    batch_size:int = 16
    lr:float=1e-3
    num_epochs:int=100
    loss_type:str='dice' # ['dice','combined']

    # develop mode
    save_gt_mask:bool=False
    isCrop: bool= True



save_gt_mask = True
dataset_type = 'ISIC2017'
isCrop = False

trainSegConfig = TrainSegConfig(dataset_type = dataset_type,
                                save_gt_mask = save_gt_mask,
                                isCrop=isCrop)
if trainSegConfig.save_gt_mask:
    print('WARNING! Saving gt mask at developing mode.')

