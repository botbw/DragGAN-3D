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
from core.log import *
from core.gen_mesh_ply import *
import math
import numpy as np
from training.volumetric_rendering import math_utils
import logging


def wrap_eg3d_backbone(backbone: TriPlaneGenerator) -> nn.Module:
    def renderer_forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
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


        return rgb_final, depth_final, weights.sum(2), sample_coordinates

    # modified from eg3d/eg3d/training/triplane.py
    def backbone_synthesis(self,
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

        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics,
                                                       neural_rendering_resolution)

        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        feature_samples, depth_samples, weights_samples, sampled_coordinates = self.renderer(
            planes, self.decoder, ray_origins, ray_directions,
            self.rendering_kwargs)  # channels last

        # RESHAPE INTO INPUT IMAGE
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H,
                                                                 W).contiguous()

        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W).squeeze()
        ray_origins_image = ray_origins.permute(0, 2, 1).reshape(N, 3, H,
                                                                 W).squeeze().permute(1, 2, 0)
        ray_directions_image = ray_directions.permute(0, 2,
                                                      1).reshape(N, 3, H,
                                                                 W).squeeze().permute(1, 2, 0)

        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(
            rgb_image,
            feature_image,
            ws,
            noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
            **{k: synthesis_kwargs[k]
               for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {
            'image': sr_image,
            'image_raw': rgb_image,
            'depth_image': depth_image,
            'planes': planes,
            'ray_origins_image': ray_origins_image[0, 0],
            'ray_directions_image': ray_directions_image,
            'sampled_coordinates': sampled_coordinates
        }

    def backbone_forward(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                neural_rendering_resolution=None,
                update_emas=False,
                cache_backbone=False,
                use_cached_backbone=False,
                **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return ws, self.synthesis(ws,
                                  c,
                                  update_emas=update_emas,
                                  neural_rendering_resolution=neural_rendering_resolution,
                                  cache_backbone=cache_backbone,
                                  use_cached_backbone=use_cached_backbone,
                                  **synthesis_kwargs)

    def sample_using_ws(self, coordinates, directions, ws, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly
        # used for extracting shapes.
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # #################
        # x, y, z, r  = 128, 128, 194, 5
        # fill_value = 1e13
        # planes[0, 0, :, y-r:y+r, x-r:x+r] = fill_value # yx
        # planes[0, 1, :, z-r:z+r, x-r:x+r] = fill_value # zx
        # planes[0, 2, :, y-r:y+r, z-r:z+r] = fill_value # yz
        # #################
        return self.renderer.run_model(planes, self.decoder, coordinates, directions,
                                       self.rendering_kwargs)

    backbone.synthesis = partial(backbone_synthesis, backbone)
    backbone.forward = partial(backbone_forward, backbone)
    backbone.renderer.forward = partial(renderer_forward, backbone.renderer)
    backbone.sample_using_ws = partial(sample_using_ws, backbone)

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
        self.forward_initialezed = False
        self.init_run = False
        self.planes0_points_ref = None
        self.planes0 = None

    def init_ws(self, z, camera_parameters):
        ws, synthesised = self.backbone_3dgan(z, camera_parameters)
        self.synthesised = synthesised
        self.init_run = True
        return ws

    def convert_pixel_to_points(self, pixel_i: int, pixel_j: int) -> WorldPoint:
        assert self.init_run, "Please call init_ws() first to get ws"
        return pixel_to_real(pixel_i, pixel_j, self.synthesised['ray_origins_image'],
                             self.synthesised['ray_directions_image'],
                             self.synthesised['depth_image'])

    def forward(self,
                ws: torch.Tensor,
                camera_parameters: torch.Tensor,
                points_cur: List[WorldPoint],
                points_target: List[WorldPoint],
                # point_step: float = 1,
                r1: float = (2 / 256) * 3, # num pixels
                r1_step: float = (2 / 256) * 1,
                r2: float = (2 / 256) * 12, # num pixels
                r2_step: float = (2 / 256) * 1,
                mask: Optional[torch.Tensor] = None):
        assert self.init_run, "Please call init_ws() first to get ws"
        assert len(points_cur) == len(points_target), "Number of points should be the same."

        synthesised = self.backbone_3dgan.synthesis(ws, camera_parameters)
        # planes (3, 32, 256, 256)
        img, planes = synthesised['image'], synthesised['planes']

        if self.planes0_points_ref is None:
            assert self.planes0 is None
            self.planes0 = planes.detach()
            points_feat = []
            for p_cur in points_cur:  # TODO @botbw: decouple from eg3d and simplify this
                feat = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes,
                    p_cur.to_tensor().unsqueeze(0).unsqueeze(0),  # _, M, _ = coordinates.shape
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
                points_feat.append(feat.reshape(96))  # feat: [1, 3, 1, 32]
            self.planes0_points_ref = torch.stack(points_feat, dim=0)

        # Point tracking with feature matching
        points_after_step = []
        move_vectors = []
        with torch.no_grad():
            for i, (point, p_tar) in enumerate(zip(points_cur, points_target)):
                coordinates = point.gen_cube_coordinates(r1, r1_step) + random.uniform(
                    -r1_step / 2, r1_step / 2) # random shift
                logging.info(
                    f'Point track: numebr of cube coordinates: {coordinates.shape[0]}, r1: {r1}, r1_step: {r1_step}'
                )
                feat_patch = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes,
                    coordinates.unsqueeze(0),  # _, M, _ = coordinates.shape
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp']).squeeze().permute(1, 0, 2).view(-1, 96)
                dir_fix = (p_tar.to_tensor() - coordinates).norm(p=2, dim=-1) * 2000
                L2 = (feat_patch - self.planes0_points_ref[i]).norm(dim=-1)
                loss = dir_fix + L2
                idx = torch.argmin(loss)
                move_vectors.append(coordinates[idx] - point.to_tensor())
                points_after_step.append(WorldPoint(coordinates[idx]))
                logging.info(f'Point track: move vector: {move_vectors[i]}, distance: {torch.norm(move_vectors[i])}')
                if torch.allclose(coordinates[idx], point.to_tensor()):
                    logging.warning(f'Point track: point {point} is not moving.')

        assert len(points_cur) == len(points_target), "Number of points should be the same."

        loss_motion = 0
        for p_cur, p_tar in zip(points_cur, points_target):
            v_cur_tar = p_tar.to_tensor() - p_cur.to_tensor()
            e_cur_tar = v_cur_tar / torch.norm(v_cur_tar)
            with torch.no_grad():
                coordinates_cur = p_cur.gen_sphere_coordinates(r2, r2_step)
                logging.info(
                    f'Motion supervision: numebr of sphere coordinates: {coordinates_cur.shape[0]}, r2: {r2}, r2_step: {r2_step}'
                )
                move_v = e_cur_tar * (2 / 256) * 30 # num pixel shift
                coordinates_step = coordinates_cur + move_v
                logging.info(f'Motion supervision: move vector: {move_v}, distance: {move_v.norm()}')
            feat_cur = sample_from_planes(
                self.backbone_3dgan.renderer.plane_axes,
                planes.detach(),
                coordinates_cur.unsqueeze(0),
                box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
            feat_step = sample_from_planes(
                self.backbone_3dgan.renderer.plane_axes,
                planes,
                coordinates_step.unsqueeze(0),
                box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
            loss_motion += F.l1_loss(feat_cur, feat_step)

        assert mask is None, "customized mask not supported yet."

        if mask is None:
            for point in points_cur:
                dis_mask = (synthesised['sampled_coordinates'] - point.to_tensor()).norm(dim=-1) < 0.05
                if mask is None:
                    mask = dis_mask
                else:
                    mask = mask | dis_mask

        masked_coord = synthesised['sampled_coordinates'][mask]
        logging.info(f'total points: {synthesised["sampled_coordinates"].shape[1]}, masked coordinates: {masked_coord.shape[0]}')
        if not masked_coord.shape[0] == 0:
            mask_loss =  F.l1_loss(
                sample_from_planes(self.backbone_3dgan.renderer.plane_axes, self.planes0, masked_coord.unsqueeze(0),
                                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp']),
                sample_from_planes(self.backbone_3dgan.renderer.plane_axes, planes, masked_coord.unsqueeze(0),
                                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
            )
            loss_motion += mask_loss
        return loss_motion, points_after_step, img, synthesised['depth_image']


if __name__ == "__main__":
    import pickle
    from eg3d.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler

    seed_everything(100861)
    setup_logger(logging.INFO)

    with open('ckpts/ffhq-fixed-triplane512-128.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    # see C_xyz at eg3d/docs/camera_coordinate_conventions.jpg
    cam2world_pose = LookAtPoseSampler.sample(
        horizontal_mean=0.5 * math.pi,  # 相机绕轴按radius旋转 0 -> pi: left view -> right view
        vertical_mean=0.5 * math.pi,  # 相机绕轴按radius 0 -> pi: up view -> down view
        lookat_position=torch.tensor(
            G.rendering_kwargs['avg_camera_pivot'],  # 按doc坐标是(x, z, y)
            device='cuda'),
        radius=G.rendering_kwargs['avg_camera_radius'],  # 相机在world[0, 0, 2.7]
        device='cuda')

    fov_deg = 18.837  # 0 -> inf : zoom-in -> zoom-out
    intrinsics = FOV_to_intrinsics(fov_deg, device='cuda')
    c = torch.cat(
        [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)  # camera parameters

    model = DragStep(wrap_eg3d_backbone(G), torch.device('cuda'))

    z = torch.randn([1, G.z_dim]).cuda()

    ws0 = model.init_ws(z, c).detach()
    ws = ws0.detach().clone().requires_grad_(True)

    gen_mesh_ply('output/mesh_start.ply', G, ws.detach(), mesh_res=256)
    opt = torch.optim.SGD([ws], lr=0.01)

    points_cur = [WorldPoint(torch.tensor([0, 0, 0.26], device='cuda'))]
    points_target = [WorldPoint(torch.tensor([0, 0, 0.5], device='cuda'))]

    try:
        for step in range(1000):
            assert torch.allclose(ws[:, 6:,:], ws0[:,6:,:])
            assert not (step != 0 and torch.allclose(ws[:, :6, :], ws0[:, :6, :]))
            ws_input = torch.cat([ws[:,:6,:], ws0[:,6:,:]], dim=1)
            loss, points_step, img, img_depth = model(ws=ws_input,
                                                    camera_parameters=c,
                                                    points_cur=points_cur,
                                                    points_target=points_target)
            logging.warn(f'points_cur: {points_cur}, points_step: {points_step}')
            points_cur = points_step
            logging.info(f'step {step} loss: {loss.item()}, ws: {ws.mean()}')
            assert not loss.isnan()
            opt.zero_grad()
            loss.backward()
            opt.step()
            break
            if step % 10 == 0:
                save_eg3d_img(img, f'output/step_{step}.png')
                # gen_mesh_ply(f'output/mesh_{step}.ply', model.backbone_3dgan, ws.detach(), mesh_res=256)
            if torch.norm(points_cur[0].to_tensor() - points_target[0].to_tensor()) < 0.1:
                break
    except KeyboardInterrupt:
        pass

    logging.info(f'points_end: {points_cur}')
    gen_mesh_ply(f'output/mesh_end.ply', model.backbone_3dgan, ws.detach(), mesh_res=256)
    print("passed")
