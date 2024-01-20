from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from functools import partial
from core.point import Point
import torch.nn.functional as F
from training.volumetric_rendering.renderer import sample_from_planes


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

    def __init__(self,
                 backbone_3dgan: GAN3DBackbone,
                 device: torch.device = torch.device('cuda'),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(
            backbone_3dgan, GAN3DBackbone
        ), "Please wrap your backbone to make sure it returns what are enfored in GAN3DBackbone."
        self.device = device
        self.backbone_3dgan = backbone_3dgan.to(device)
        self.w0 = None
        self.planes0_ref = None
        self.planes0_resized = None
        self.device = device

    # # Generate random latents.
    # z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

    # # Run mapping network.
    # label = torch.zeros([1, G.c_dim], device=self._device)
    # w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
    def forward(
        self,
        ws: torch.Tensor,
        camera_parameters: torch.Tensor,
        points_cur: List[Point],
        points_target: List[Point],
        mask: Optional[torch.Tensor]=None,
        r1: int=3,
        r2: int=12,
    ):
        # if self.w0 is None:  # TODO @botbw: try to make forward functional
        #     self.w0 = ws.detach().clone()

        # ws = torch.cat([ws[:, :6, :], self.w0[:, 6:, :]], dim=1)

        ret = self.backbone_3dgan.foward(ws, camera_parameters)
        img, planes = ret['image'], ret['planes']
        assert planes.shape == (
            1, 3, 32, 256, 256
        ), "eg3d plane sizes"  # TODO @botbw: decouple from eg3d, remove redundant dims
        planes = planes[0]
        assert planes.shape == (3, 32, 256, 256), "eg3d plane sizes"
        # h, w = G.img_resolution, G.img_resolution
        x_lim, y_lim, z_lim = 300, 300, 300  # TODO @botbw should be output figure size
        # H = W = self.neural_rendering_resolution?

        planes_resized = F.interpolate(planes, [
            x_lim, y_lim
        ], mode='bilinear').unsqueeze(
            0
        )  # TODO @botbw: decouple from eg3d, requires (N, n_planes, C, H, W)

        if self.planes0_ref is None:
            assert self.planes0_resized is None
            self.planes0_resized = planes_resized.detach().clone()
            points_feat = []
            for point in points_cur:  # TODO @botbw: decouple from eg3d and simplify this
                feat = sample_from_planes(self.backbone_3dgan.backbone.renderer.plane_axes,
                                       self.planes0_resized,
                                       point.to_tensor(
                                           self.device).unsqueeze(0).unsqueeze(0), # _, M, _ = coordinates.shape
                                       box_warp=self.backbone_3dgan.backbone.    # batch (1), n_points, 3
                                       rendering_kwargs['box_warp'])
                points_feat.append(feat) # feat: [1, 3, 1, 32]
            self.planes0_ref = torch.concat(points_feat, dim=2).permute(2, 0, 1, 3).reshape(-1, 96) # [1, 3, n_points, 32] to [n_points, 96]

        # Point tracking with feature matching
        points_after_step = []
        with torch.no_grad():
            for i, point in enumerate(points_cur):
                r = round(r2 / 512 * x_lim) # TODO @botbw: what is this? is r2 / 512 relative value?
                cordinates = point.gen_cube_coordinates(r, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim).unsqueeze(0).to(self.device)
                feat_patch = sample_from_planes(self.backbone_3dgan.backbone.renderer.plane_axes, planes_resized, cordinates, box_warp=self.backbone_3dgan.backbone.rendering_kwargs['box_warp']) # [1, 3, n_points, 32]
                feat_patch = feat_patch.squeeze().permute(1, 0, 2).view(-1, 96) # [1, 3, n_points, 32] to [n_points, 96]
                L2 = torch.linalg.norm(feat_patch - self.planes0_ref[i], dim=-1)
                _, idx = torch.min(L2.view(1,-1), -1)
                points_after_step.append(Point(cordinates[0, idx, 0], cordinates[0, idx, 1], cordinates[0, idx, 2]))

        assert len(points_cur) == len(points_target), "Number of points should be the same."

        finished = True # TODO @botbw: change to tensor
        for p_cur, p_tar in zip(points_cur, points_target):
            v_cur_tar = p_tar.to_tensor(self.device) - p_cur.to_tensor(self.device)
            length_v_cur_tar = torch.norm(v_cur_tar)

            if length_v_cur_tar > max(2 / 512 * x_lim, 2): # TODO @botbw: update this according to eg3d
                finished = False

            if length_v_cur_tar > 1:
                r = round(r1 / 512 * x_lim)
                e_cur_tar = v_cur_tar / length_v_cur_tar
                coordinates = p_cur.gen_sphere_coordinates(r, self.device, x_lim, y_lim, z_lim)
                normed_new_coordinates = (coordinates + e_cur_tar) / torch.tensor([x_lim, y_lim, z_lim], device=self.device) * 2 - 1



if __name__ == "__main__":
    import pickle
    from eg3d.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2,
                                            3.14 / 2,
                                            torch.tensor([0, 0, 0.2],
                                                        device='cuda'),
                                            radius=2.7,
                                            device='cuda')

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device='cuda')
    with open('ckpts/ffhq512-128.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    z = torch.randn([1, G.z_dim]).cuda()    # latent codes
    c = torch.cat([cam2world_pose.reshape(-1, 16),
                intrinsics.reshape(-1, 9)], 1)  # camera parameters
    DragStep(wrap_eg3d_backbone(G), torch.device('cuda'))(
        ws=z,
        camera_parameters=c,
        points_cur=[Point(0, 0, 0), Point(1, 1, 1)],
        points_target=[Point(10, 10, 10), Point(20, 20, 20)],
    )