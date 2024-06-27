import os
import pprint
from dataclasses import asdict
import numpy as np
import random

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0 

def get_version_num(path_dir:str):
    '''
    get the version number given a folder path 
    '''
    if os.path.isdir(path_dir):
        # Get all folders numbers in the root_log_dir
        version_started_folder_names = [each for each in os.listdir(path_dir) if each.startswith("version_") ]
        # print(version_started_folder_names)
        folder_numbers = [int(folder.replace("version_", "")) for folder in version_started_folder_names]
        
        if len(folder_numbers) == 0:
            version_num = 0
        else:
            # Find the latest version number present in the log_dir
            version_num = max(folder_numbers) + 1
    else:
        version_num = 0

    return str(version_num)


def save_config(config,
                save_path: str):
    config_str = pprint.pformat(asdict(config),sort_dicts=False)
    with open(save_path, "w") as f:
        f.write(config_str)
    return


def find_random_point_in_mask(mask,fix_middle_point=False):
    if fix_middle_point:
        h,w = mask.size
        return h//2,w//2
    
    # Find all non-zero points in the mask
    # print(f'{mask.size=}')
    y_indices, x_indices = np.nonzero(mask)
    if len(y_indices) == 0:
        raise ValueError("The mask contains no non-zero points.")
    # Randomly select one point
    idx = random.randint(0, len(y_indices) - 1)
    return x_indices[idx], y_indices[idx]

def crop_image_at_point(image, mask,point,random_seed=None):
    x, y = point
    w,h = image.size

    # print(f'{x=},{y=},{h=},{w=}')

    if random_seed is not None:
        random.seed(random_seed)

    quadrant_number = random.randint(1, 4)
    if quadrant_number == 1:
        # Top-left
        return image.crop((0, 0, x, y)), mask.crop((0, 0, x, y))
    elif quadrant_number == 2:
        # Top-right
        return image.crop((x, 0, w, y)), mask.crop((x, 0, w, y))
    elif quadrant_number == 3:
        # Bottom-left
        return image.crop((0, y, x, h)), mask.crop((0, y, x, h))
    elif quadrant_number == 4:
        # Bottom-right
        return image.crop((x, y, w, h)), mask.crop((x, y, w, h))
    else:
        raise ValueError("Quadrant number must be between 1 and 4.")
    




