import torch


@torch.no_grad
def norm_img_depth(img_depth, rendering_kwargs):
    img_depth = rendering_kwargs['avg_camera_radius'] - img_depth
    return img_depth
