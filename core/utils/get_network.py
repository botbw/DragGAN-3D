from typing import Union
import os
from core import PROJ_DIR

import pickle


def get_network(pkl_file_path: Union[str, os.PathLike]) -> dict:
    """Get a StyleGAN network from a pkl file."""
    assert os.path.exists(pkl_file_path), f'pkl file {pkl_file_path} does not exist!'
    with open(pkl_file_path, 'rb') as f:
        obj = pickle.load(f)  # torch.nn.Module
    return obj


if __name__ == '__main__':
    tmp = get_network(f'{PROJ_DIR}/ckpts/ffhq512-128.pkl')
    print("passed")
