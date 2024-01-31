from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from functools import partial
from core.point import *
import torch.nn.functional as F
from eg3d.eg3d.training.triplane import TriPlaneGenerator
from training.volumetric_rendering.renderer import sample_from_planes, project_onto_planes
from core.utils import *
import math
import numpy as np
from training.volumetric_rendering import math_utils

def wrap_eg3d_backbone(backbone: TriPlaneGenerator) -> nn.Module:
    def forward_renderer(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        return rgb_final, depth_final, weights.sum(2), sample_coordinates.detach().reshape(batch_size, num_rays, samples_per_ray, 3)

    backbone.renderer.forward = partial(forward_renderer, backbone.renderer)

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

        # 0: 人物正面视角左下为原点, xz平面， 1人物上面视角右上为原点，xy平面
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])
        #################
        # p = Point(128, 128, 128)
        # cor = p.gen_cube_coordinates(30, 256, 256 ,256).unsqueeze(0).cuda().float()
        # cor = project_onto_planes(self.renderer.plane_axes, cor).long()
        # i_range = torch.arange(3)
        # j_range = torch.arange(cor.shape[1])

        # i_grid, _ = torch.meshgrid(i_range, j_range)
        # planes[0, i_grid, :, cor[:, :, 0], cor[:, :, 1]] = 1e-9
        # for i in range(3):
        #     for j in range(cor.shape[1]):
        #         planes[0, i, :, cor[i, j, 0], cor[i, j, 1]] = 0
        #################
        feature_samples, depth_samples, weights_samples, sample_coordinates_raw = self.renderer(
            planes, self.decoder, ray_origins, ray_directions,
            self.rendering_kwargs)  # channels last

        # RESHAPE INTO INPUT IMAGE
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()

        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W).squeeze()
        ray_origins_image = ray_origins.permute(0, 2, 1).reshape(N, 3, H, W).squeeze().permute(1, 2, 0)
        ray_directions_image = ray_directions.permute(0, 2, 1).reshape(N, 3, H, W).squeeze().permute(1, 2, 0)

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
            'depth_image': depth_image,
            'planes': planes,
            'ray_origins_image': ray_origins_image[0, 0],
            'ray_directions_image': ray_directions_image,
            'sample_coordinates': sample_coordinates_raw.reshape(H, W, self.rendering_kwargs['depth_resolution'], 3)
        }

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return ws, self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

    backbone.synthesis = partial(synthesis, backbone)
    backbone.forward = partial(forward, backbone)

    return backbone


class DragStep(nn.Module):

    def __init__(self,
                 backbone_3dgan: TriPlaneGenerator,
                 device: torch.device = torch.device('cuda'),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.backbone_3dgan = backbone_3dgan.to(device)
        self.w0 = None
        self.planes0_ref = None
        self.planes0_resized = None
        self.device = device

        # for debug
        self.feat_his = []

    def init_forward(self, z, camera_parameters):
        return self.backbone_3dgan(z, camera_parameters)

    def forward(
        self,
        ws: torch.Tensor,
        camera_parameters: torch.Tensor,
        points_cur: List[FeatPoint],
        points_target: List[FeatPoint],
        mask: Optional[torch.Tensor]=None,
        r1: int=3,
        r2: int=12,
    ):
        synthesised = self.backbone_3dgan.synthesis(ws, camera_parameters)
        img, planes = synthesised['image'], synthesised['planes']
        assert planes.shape == (
            1, 3, 32, 256, 256
        ), "eg3d plane sizes"
        planes = planes[0]
        assert planes.shape == (3, 32, 256, 256), "eg3d plane sizes"
        # h, w = G.img_resolution, G.img_resolution
        x_lim, y_lim, z_lim = 256, 256, 256
        # # H = W = self.neural_rendering_resolution?

        # planes_resized = F.interpolate(planes, [
        #     x_lim, y_lim
        # ], mode='bilinear').unsqueeze(
        #     0
        # )
        self.feat_his.append(planes.detach().clone())

        if self.planes0_ref is None:
            assert self.planes0_resized is None
            self.planes0_resized = planes.detach().clone().unsqueeze(0)
            points_feat = []
            for point in points_cur:  # TODO @botbw: decouple from eg3d and simplify this
                feat = sample_from_planes(
                        self.backbone_3dgan.renderer.plane_axes,
                        self.planes0_resized,
                        point.to_real(self.device).unsqueeze(0).unsqueeze(0), # _, M, _ = coordinates.shape
                        box_warp=self.backbone_3dgan.    # batch (1), n_points, 3
                        rendering_kwargs['box_warp']
                    )
                points_feat.append(feat.reshape(96)) # feat: [1, 3, 1, 32]
            self.planes0_ref = torch.stack(points_feat, dim=0)

        # Point tracking with feature matching
        points_after_step = []
        with torch.no_grad():
            for i, point in enumerate(points_cur):
                r = round(r2 / 512 * x_lim) # TODO @botbw: what is this? is r2 / 512 relative value?
                coordinates = point.gen_real_cube_coordinates(r, device=self.device)
                feat_patch = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes.unsqueeze(0),
                    coordinates.unsqueeze(0), # _, M, _ = coordinates.shape
                    box_warp=self.backbone_3dgan.    # batch (1), n_points, 3
                    rendering_kwargs['box_warp']
                )
                # feat_patch = torch.nn.functional.grid_sample(planes, coordinates, mode='bilinear', padding_mode='zeros', align_corners=False).permute(2, 3, 0, 1).squeeze(0)
                feat_patch = feat_patch.squeeze().permute(1, 0, 2).view(-1, 96) # [1, 3, n_points, 32] to [n_points, 96]
                L2 = torch.linalg.norm(feat_patch - self.planes0_ref[i], dim=-1)
                idx = torch.argmin(L2.view(1,-1), -1).item()
                points_after_step.append(
                    real_to_feat(coordinates[idx], resolution=256)
                )

        assert len(points_cur) == len(points_target), "Number of points should be the same."

        finished = True # TODO @botbw: change to tensor
        loss_motion = 0
        for p_cur, p_tar in zip(points_cur, points_target):
            v_cur_tar = p_tar.to_tensor(self.device) - p_cur.to_tensor(self.device)
            length_v_cur_tar = torch.norm(v_cur_tar)

            if length_v_cur_tar > max(2 / 512 * x_lim, 2): # TODO @botbw: update this according to eg3d
                finished = False

            if length_v_cur_tar > 1:
                r = round(r1 / 512 * x_lim)
                e_cur_tar = v_cur_tar / length_v_cur_tar
                coordinates_cur = p_cur.gen_real_sphere_coordinates(r, device=self.device).unsqueeze(0)
                p_step = tensor_to_feat(p_cur.to_tensor(self.device) + e_cur_tar, resolution=256)
                coordinates_step = p_step.gen_real_sphere_coordinates(r, device=self.device).unsqueeze(0)
                feat_cur = sample_from_planes(self.backbone_3dgan.renderer.plane_axes,
                                       planes.unsqueeze(0),
                                       coordinates_cur, # _, M, _ = coordinates.shape
                                       box_warp=self.backbone_3dgan.    # batch (1), n_points, 3
                                       rendering_kwargs['box_warp']).detach()
                feat_step = sample_from_planes(self.backbone_3dgan.renderer.plane_axes,
                                        planes.unsqueeze(0),
                                        coordinates_step, # _, M, _ = coordinates.shape
                                        box_warp=self.backbone_3dgan.    # batch (1), n_points, 3
                                        rendering_kwargs['box_warp'])
                loss_motion += F.l1_loss(feat_cur, feat_step)

        return loss_motion, points_after_step, img, synthesised['depth_image']


if __name__ == "__main__":
    import pickle
    from eg3d.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler

    seed_everything(100861)

    with open('ckpts/ffhq-fixed-triplane512-128.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    # see C_xyz at eg3d/docs/camera_coordinate_conventions.jpg
    cam2world_pose = LookAtPoseSampler.sample(horizontal_mean=0.5 * math.pi, # 相机绕轴按radius旋转 0 -> pi: left view -> right view
                                            vertical_mean=0.5 * math.pi, # 相机绕轴按radius 0 -> pi: up view -> down view
                                            lookat_position=torch.tensor(G.rendering_kwargs['avg_camera_pivot'], # 按doc坐标是(x, z, y)
                                                        device='cuda'),
                                            radius=G.rendering_kwargs['avg_camera_radius'], # 相机在world[0, 0, 2.7]
                                            device='cuda')

    fov_deg = 18.837 # 0 -> inf : zoom-in -> zoom-out
    intrinsics = FOV_to_intrinsics(fov_deg, device='cuda')
    c = torch.cat([cam2world_pose.reshape(-1, 16),
                intrinsics.reshape(-1, 9)], 1)  # camera parameters


    model = DragStep(wrap_eg3d_backbone(G), torch.device('cuda'))
    z = torch.randn([1, G.z_dim]).cuda().requires_grad_(True)    # latent codes
    ws, synthesised = model.init_forward(z, c)
    ws = ws.detach().requires_grad_(True)
    opt = torch.optim.SGD([ws], lr=0.01)
    pixel_i, pixel_j = 0, 0

    point_real = pixel_to_real(pixel_i, pixel_j, synthesised['ray_origins_image'], synthesised['ray_directions_image'], synthesised['depth_image'], device='cuda')
    points_cur = [real_to_feat(point_real, resolution=256)]

    for step in range(10):
        loss, points_step, img, img_depth = model(
            ws=ws,
            camera_parameters=c,
            points_cur=points_cur,
            points_target=[FeatPoint(256, 256, 256)]
        )
        # save_3d_depth_img(img_depth)
        print(f'points_cur: {points_cur}, points_step: {points_step}')
        points_cur = points_step
        print(f'loss: {loss.item()}, ws: {ws.mean()}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        save_eg3d_img(img, f'step_{step}.png')
    print("passed")