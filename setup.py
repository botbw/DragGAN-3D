from setuptools import setup, find_packages

setup(
    name='DragGAN-3D',
    version='0',
    packages=find_packages(),
)

import os
from core import PROJ_DIR

if not os.path.exists(f'{PROJ_DIR}/dnnlib'):
    os.system(f'ln -s {PROJ_DIR}/eg3d/eg3d/dnnlib {PROJ_DIR}/dnnlib')
if not os.path.exists(f'{PROJ_DIR}/torch_utils'):
    os.system(f'ln -s {PROJ_DIR}/eg3d/eg3d/torch_utils {PROJ_DIR}/torch_utils')
if not os.path.exists(f'{PROJ_DIR}/training'):
    os.system(f'ln -s {PROJ_DIR}/eg3d/eg3d/training {PROJ_DIR}/training')
