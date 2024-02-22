from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from functools import partial
from core.world_point import *
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
from core.feat_point import FeatPoint, sample


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
        # x, y, z, r  = 128, 128, 200, 3
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
        assert self.init_run, "Call init_ws() first to get ws"
        return pixel_to_real(pixel_i, pixel_j, self.synthesised['ray_origins_image'],
                             self.synthesised['ray_directions_image'],
                             self.synthesised['depth_image'])

    def forward(self,
                ws: torch.Tensor,
                camera_parameters: torch.Tensor,
                points_cur: List[FeatPoint],
                points_target: List[FeatPoint],
                point_step: float = 1,
                r1: int = 3, # num pixels
                r2: int = 3, # num pixels
                mask_loss_lambda: float = 1.0,
                mask: Optional[torch.Tensor] = None):
        assert self.init_run, "Please call init_ws() first to get ws_0"
        assert len(points_cur) == len(points_target), "Number of points should be the same."

        synthesised = self.backbone_3dgan.synthesis(ws, camera_parameters)
        img, planes = synthesised['image'], synthesised['planes']

        if self.planes0_points_ref is None:
            assert self.planes0 is None
            self.planes0 = planes.detach()
            points_feat = []
            for p_cur in points_cur:
                feat = sample(planes, p_cur.to_tensor().unsqueeze(0))
                points_feat.append(feat.reshape(96))
            self.planes0_points_ref = torch.stack(points_feat, dim=0)

        # Point tracking with feature matching
        points_after_step = []
        with torch.no_grad():
            for i, (p_cur, p_tar) in enumerate(zip(points_cur, points_target)):
                coordinates = p_cur.gen_cube_coordinates(r1)
                logging.info(
                    f'Point track: numebr of cube coordinates: {coordinates.shape[0]} '#, r1: {r1}, r1_step: {r1_step}'
                )
                feat_patch = sample(planes, coordinates)
                loss = (feat_patch - self.planes0_points_ref[i]).norm(dim=-1)
                idx = torch.argmin(loss)
                move_vector = (coordinates[idx].float() - p_cur.to_tensor())
                points_after_step.append(FeatPoint(coordinates[idx][0].item(), coordinates[idx][1].item(), coordinates[idx][2].item()))
                logging.info(f'Point track: move vector: {move_vector}, distance: {move_vector.float().norm()}')
                if torch.allclose(coordinates[idx], p_cur.to_tensor()):
                    logging.warning(f'Point track: point {p_cur} is not moving.')

        loss = 0

        loss_motion = 0
        for p_cur, p_tar in zip(points_cur, points_target):
            with torch.no_grad():
                v_cur_tar = p_tar.to_tensor() - p_cur.to_tensor()
                e_cur_tar = v_cur_tar / (torch.norm(v_cur_tar.float()) + 1e-9)
                coordinates_cur = p_cur.gen_sphere_coordinates(r2)
                logging.info(
                    f'Motion supervision: numebr of sphere coordinates: {coordinates_cur.shape[0]},' # r2: {r2}, r2_step: {r2_step}'
                )
                coordinates_step = torch.round(coordinates_cur + e_cur_tar * point_step).long().clamp_max(255)
                feat_cur = sample(planes, coordinates_cur) # actually no need to detach again
            feat_step = sample(planes, coordinates_step)
            loss_motion += F.l1_loss(feat_cur.detach(), feat_step)

        loss += loss_motion

        mask_loss = 0
        for p_cur, p_tar in zip(points_cur, points_target):
            with torch.no_grad():
                dis_mask = torch.ones(3, 256, 256, dtype=torch.bool, device=self.device)
                p_mid_tensor = (0.5 * (p_cur.to_tensor().float() + p_tar.to_tensor())).round().int()
                p_mid = FeatPoint(p_mid_tensor[0].item(), p_mid_tensor[1].item(), p_mid_tensor[2].item())
                R = max(1, (p_tar.to_tensor().float() - p_cur.to_tensor()).norm().round().int().item() // 2)
                coordinates = p_mid.gen_cube_coordinates(R)
                dis_mask[0, coordinates[..., 1], coordinates[..., 0]] = 0
                dis_mask[1, coordinates[..., 2], coordinates[..., 0]] = 0
                dis_mask[2, coordinates[..., 1], coordinates[..., 2]] = 0
                if mask is None:
                    mask = dis_mask
                else:
                    mask = mask & dis_mask
            mask_loss += mask_loss_lambda * F.l1_loss(planes.permute(0, 2, 1, 3, 4) * mask, self.planes0.permute(0, 2, 1, 3, 4) * mask)

        loss += mask_loss
        logging.info(f'loss: {loss.item()}, motion_loss: {loss_motion.item()}, mask_loss: {mask_loss.item()}')

        return loss, points_after_step, img, synthesised['depth_image']


if __name__ == "__main__":
    import pickle
    from eg3d.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
    seed = 100861
    seed_everything(seed)
    setup_logger(logging.INFO)

    with open('ckpts/ffhq-fixed-triplane512-128.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    # see C_xyz at eg3d/docs/camera_coordinate_conventions.jpg
    cam2world_pose = LookAtPoseSampler.sample(
        horizontal_mean=0.5 * math.pi,
        vertical_mean=0.5 * math.pi,
        lookat_position=torch.tensor(
            G.rendering_kwargs['avg_camera_pivot'],
            device='cuda'),
        radius=G.rendering_kwargs['avg_camera_radius'],
        device='cuda')

    fov_deg = 18.837  # 0 -> inf : zoom-in -> zoom-out
    intrinsics = FOV_to_intrinsics(fov_deg, device='cuda')
    c = torch.cat(
        [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)  # camera parameters

    model = DragStep(wrap_eg3d_backbone(G), torch.device('cuda'))

    z = torch.randn([1, G.z_dim]).cuda()

    ws0 = model.init_ws(z, c).detach()
    ws = ws0.detach().clone().requires_grad_(True)
    print(f'ws0: {ws0.shape}')

    gen_mesh_ply(f'output/{seed}_start.ply', G, ws.detach(), mesh_res=256)
    opt = torch.optim.SGD([ws], lr=0.001)
    
    def convert(x, y, z, box_wrap=G.rendering_kwargs['box_warp']):
        # (2 / box_wrap * (x, y, z) + 1) / 2 * 256 (plane resolution)
        return (
            round((2 / box_wrap * x + 1) / 2 * 256),
            round((2 / box_wrap * y + 1) / 2 * 256),
            round((2 / box_wrap * z + 1) / 2 * 256)
        )

    # 移动鼻子
    # p = FeatPoint(*convert(0, 0.016565, 0.263870))
    # t_p = FeatPoint(p.x, p.y - 10, p.z)
    # points_cur = [p]
    # points_target = [t_p]

    # # 移动眼睛
    p1 = FeatPoint(*convert(0.08, 0.0963, 0.201))
    p2 = FeatPoint(*convert(0.08, 0.0563, 0.191))
    p1_t = FeatPoint(p1.x, p1.y - 10, p1.z)
    p2_t = FeatPoint(p2.x, p2.y + 10, p2.z)
    points_cur = [p1, p2]
    points_target = [p1_t, p2_t]

    # 移动嘴巴
    # p1 = FeatPoint(*convert(0.0567, -0.08, 0.206))
    # p2 = FeatPoint(*convert(-0.0567, -0.08, 0.206))
    # p1_t = FeatPoint(p1.x + 5, p1.y, p1.z)
    # p2_t = FeatPoint(p2.x - 5, p2.y, p2.z)
    # points_cur = [p1, p2]
    # points_target = [p1_t, p2_t]

    #嘴唇
    # p1 = FeatPoint(*convert(0.004924, -0.064737, 0.267408))
    # p1_t = FeatPoint(p1.x, p1.y + 5, p1.z)
    # points_cur = [p1]
    # points_target = [p1_t]

    # # 移动头发/帽子
    # p_le = FeatPoint(*convert(-0.1, 0.25, 0.202))
    # p_mi = FeatPoint(*convert(0, 0.25, 0.202))
    # p_ri = FeatPoint(*convert(0.1, 0.25, 0.202))
    # p_le_t = FeatPoint(p_le.x, p_le.y + 8, p_le.z-3)
    # p_mi_t = FeatPoint(p_mi.x, p_mi.y + 5, p_mi.z-3)
    # p_ri_t = FeatPoint(p_ri.x, p_ri.y + 5, p_ri.z-3)
    # points_cur = [p_le, p_mi, p_ri]
    # points_target = [p_le_t, p_mi_t, p_ri_t]

    l_w = 14
    try:
        for step in range(1500):
            assert torch.allclose(ws[:, l_w:,:], ws0[:,l_w:,:])
            assert not (step != 0 and l_w != 0 and torch.allclose(ws[:, :l_w, :], ws0[:, :l_w, :]))
            ws_input = torch.cat([ws[:,:l_w,:], ws0[:,l_w:,:]], dim=1)
            loss, points_step, img, img_depth = model(ws=ws_input,
                                                    camera_parameters=c,
                                                    points_cur=points_cur,
                                                    points_target=points_target)
            logging.info(f'step: {step}\npoints_cur: {points_cur}\npoints_step: {points_step}')
            points_cur = points_step
            assert not loss.isnan()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0:
                save_eg3d_img(img, f'output/step_{step}.png')
                # gen_mesh_ply(f'output/mesh_{step}.ply', model.backbone_3dgan, ws.detach(), mesh_res=256)
            if torch.norm((points_cur[0].to_tensor() - points_target[0].to_tensor()).float()) <= 2.0:
                break
    except KeyboardInterrupt:
        pass

    logging.info(f'points_end: {points_cur}')
    gen_mesh_ply(f'output/{seed}_end.ply', model.backbone_3dgan, ws.detach(), mesh_res=256)
    print("passed")
