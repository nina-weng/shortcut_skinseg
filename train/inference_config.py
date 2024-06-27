from dataclasses import dataclass


@dataclass
class InferenceSegConfig:
    dataset_type: str  #['ISIC2016', 'ISIC2017']
    isCrop: bool
    # load model from
    load_model_type: str
    load_model_version: int
    load_model_path: str
    # develop mode
    save_gt_mask:bool=False
    # resize to
    resized_size: tuple[int] = (256,256)
    # batch size
    batch_size:int = 16
    loss_type:str='dice' # as metrics
    #
    random_seed:int =2024
    

    


dataset_type = 'ISIC2017' #['ISIC2016', 'ISIC2017']
isCrop = True
save_gt_mask = True
load_model_type = 'unet'
load_model_version = 3 # 6
load_model_path = f'/records/trained_models/{dataset_type}/{load_model_type}/version_{str(load_model_version)}/{load_model_type}_model.pth'


inferSegConfig = InferenceSegConfig(dataset_type=dataset_type,
                                    isCrop = isCrop,
                                    save_gt_mask=save_gt_mask,
                                    load_model_type=load_model_type,
                                    load_model_version = load_model_version,
                                    load_model_path = load_model_path,
                                    )
if inferSegConfig.save_gt_mask:
    print('WARNING! Saving gt mask at developing mode.')

