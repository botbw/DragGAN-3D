import os

__cwd = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.abspath(os.path.join(__cwd, os.pardir))
CKPTS_DIR = PROJ_DIR + '/ckpts'

