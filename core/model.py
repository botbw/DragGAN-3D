from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from functools import partial
from core.point import Point
import torch.nn.functional as F


class GAN3DBackbone(nn.Module):

    def __init__(self, backbone: Callable, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = backbone

    def check_return(self, ret):
        assert isinstance(ret, dict), "Please return a dict."
        assert "image" in ret.keys(), "Please return a dict with key 'image'."
        assert "image_raw" in ret.keys(
        ), "Please return a dict with key 'image_raw'."
        assert "image_depth" in ret.keys(
        ), "Please return a dict with key 'image_depth'."
        assert "planes" in ret.keys(
        ), "Please return a dict with key 'planes'."

    def foward(self, *args, **kwargs) -> Tuple[torch.Tensor]:
        ret = self.backbone(*args, **kwargs)
        self.check_return(ret)
        return ret


def wrap_eg3d_backbone(backbone: Callable) -> GAN3DBackbone:
    # modified from eg3d/eg3d/training/triplane.py
    def synthesis(self,
                  ws,
                  c,
                  neural_rendering_resolution=None,
                  update_emas=False,
                  cache_backbone=False,
                  use_cached_backbone=False,
                  **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # assert neural_rendering_resolution is not None
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws,
                                             update_emas=update_emas,
                                             **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])
        feature_samples, depth_samples, weights_samples = self.renderer(
            planes, self.decoder, ray_origins, ray_directions,
            self.rendering_kwargs)  # channels last

        # RESHAPE INTO INPUT IMAGE
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(
            rgb_image,
            feature_image,
            ws,
            noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
            **{
                k: synthesis_kwargs[k]
                for k in synthesis_kwargs.keys() if k != 'noise_mode'
            })

        return {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'planes': planes
        }

    backbone.synthesis = partial(synthesis, backbone)

    return GAN3DBackbone(backbone)

class DragStep(nn.Module):
    def __init__(self, backbone_3dgan: GAN3DBackbone, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(backbone_3dgan, GAN3DBackbone), "Please wrap your backbone to make sure it returns what are enfored in GAN3DBackbone."
        self.backbone_3dgan = backbone_3dgan
        self.w0 = None
        self.planes0_ref = None

    # # Generate random latents.
    # z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

    # # Run mapping network.
    # label = torch.zeros([1, G.c_dim], device=self._device)
    # w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
    def forward(self,
                ws: torch.Tensor,
                camera_parameters: torch.Tensor,
                points_start: List[Point],
                points_end: List[Point],
                mask: torch.Tensor,
                ):
        if self.w0 is None: # TODO @botbw: try to make forward functional
            self.w0 = ws.detach().clone()

        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)

        ret = self.backbone_3dgan.foward(ws, camera_parameters)
        img, planes = ret['image'], ret['planes']
        assert planes.shape == (1, 3, 256, 256, 32), "eg3d plane sizes" # TODO @botbw: decouple from eg3d, remove redundant dims
        planes = planes[0].permute(0, 3, 1, 2)

        # h, w = G.img_resolution, G.img_resolution
        h, w = 300, 300 # TODO @botbw should be output figure size
                        # H = W = self.neural_rendering_resolution?


        X = torch.linspace(0, h, h)
        Y = torch.linspace(0, w, w)
        xx, yy = torch.meshgrid(X, Y, indexing='ij')
        planes = F.interpolate(planes, [h, w], mode='bilinear')

        if self.planes0_ref is None:


