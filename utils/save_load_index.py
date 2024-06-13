import torch
import numpy as np

def save_cv_bd_index(bd_index: torch.tensor,
                     cv_index: torch.Tensor,
                     save_path: str,
                     index_name: str):
    save_dict = {
        "bd_index": bd_index,
        "cv_index": cv_index,
    }

    torch.save(
        save_dict,
        f'{save_path}/{index_name}',
    )
    
def load_cv_bd_index(save_path: str):
    load_file = torch.load(save_path)
    assert all(key in load_file for key in ['bd_index', 'cv_index'])
    print(f"loading index information...")
    return load_file
