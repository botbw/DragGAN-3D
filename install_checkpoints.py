# link downloaded checkpoints to the correct location
import os

EG3D_HINT = """\
EG3D checkpoints not found, you can download it by 'wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/zip -O eg3d_1.zip' (visit https://github.com/NVlabs/eg3d/tree/main for more information).
Note that the checkpoints provided by offical repo are using buggy triplanes (see issue: https://github.com/NVlabs/eg3d/issues/67), you can retrieve some of the fixed versions from https://drive.google.com/drive/folders/1eJrXvda9ZwA8NYOLtvr4N-iJ1u9wZ27J (Provided by https://github.com/oneThousand1000/LPFF-dataset/tree/master/networks)
"""

PREPROCESSING_MODEL_HINT = """\
Data preprocessing of eg3d depends Deep3DFaceRecon_pytorch, please check https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21?tab=readme-ov-file#prepare-prerequisite-models and download all the necessary files.
"""
FILE_CONFIG = {
    'afhqcats512-128.pkl': (EG3D_HINT, None),
    'ffhq512-128.pkl': (EG3D_HINT, None),
    'ffhqrebalanced512-64.pkl': (EG3D_HINT, None),
    'ffhqrebalanced512-128.pkl': (EG3D_HINT, 'in-n-out/eg3d/pretrained_models/ffhqrebalanced512-128.pkl'),
    'shapenetcars128-64.pkl': (EG3D_HINT, None),
    'PublicMM1/01_MorphableModel.mat': (PREPROCESSING_MODEL_HINT, 'in-n-out/data_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM/01_MorphableModel.mat'),
    'Exp_Pca.bin': (PREPROCESSING_MODEL_HINT, 'in-n-out/data_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM/Exp_Pca.bin')
}

if __name__ == "__main__":
    for name, (hint, link_path) in FILE_CONFIG.items():
        path = 'ckpts/' + name
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found, {hint}")
        real_path = os.path.realpath(path)
        if link_path and (not os.path.exists(link_path)):
            if not os.path.exist(os.readlink(link_path)):
                os.remove(link_path)
            os.symlink(real_path, link_path)