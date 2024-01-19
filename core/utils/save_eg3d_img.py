import torch
from torchvision.utils import save_image

def save_eg3d_img(raw_img: torch.Tensor, path: str='my_image.jpg'):
    raw_img = raw_img.detach()
    raw_img = (raw_img + 1) / 2
    save_image(raw_img, path)
