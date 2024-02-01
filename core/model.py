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

        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics,
                                                       neural_rendering_resolution)

        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # 0: 人物正面视角左下为原点, xz平面， 1人物上面视角右上为原点，xy平面
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
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
        feature_samples, depth_samples, weights_samples = self.renderer(
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
        }

    def forward(self,
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
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    backbone.synthesis = partial(synthesis, backbone)
    backbone.forward = partial(forward, backbone)
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
        self.planes0_ref = None

        self.plane_his = []

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
                point_step: float = 0.01,
                r1: float = 0.1,
                r1_step: float = 0.05,
                r2: float = 0.1,
                r2_step: float = 0.05,
                mask: Optional[torch.Tensor] = None):
        assert self.init_run, "Please call init_ws() first to get ws"
        assert len(points_cur) == len(points_target), "Number of points should be the same."

        synthesised = self.backbone_3dgan.synthesis(ws, camera_parameters)
        img, planes = synthesised['image'], synthesised['planes']  # planes (3, 32, 256, 256)

        if self.planes0_ref is None:
            points_feat = []
            for point in points_cur:  # TODO @botbw: decouple from eg3d and simplify this
                feat = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes,
                    point.to_tensor().unsqueeze(0).unsqueeze(0),  # _, M, _ = coordinates.shape
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
                points_feat.append(feat.reshape(96))  # feat: [1, 3, 1, 32]
            self.planes0_ref = torch.stack(points_feat, dim=0)

        self.plane_his.append(planes.detach())
        # Point tracking with feature matching
        points_after_step = []
        with torch.no_grad():
            for i, point in enumerate(points_cur):
                coordinates = point.gen_real_cube_coordinates(r1, r1_step) + random.uniform(-r1_step / 2, r1_step/2)
                logging.info(f'Numebr of cube coordinates: {coordinates.shape[0]}, r1: {r1}, r1_step: {r1_step}')
                feat_patch = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes,
                    coordinates.unsqueeze(0),  # _, M, _ = coordinates.shape
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
                # feat_patch = torch.nn.functional.grid_sample(planes, coordinates, mode='bilinear', padding_mode='zeros', align_corners=False).permute(2, 3, 0, 1).squeeze(0)
                feat_patch = feat_patch.squeeze().permute(1, 0, 2).view(
                    -1, 96)  # [1, 3, n_points, 32] to [n_points, 96]
                L2 = torch.linalg.norm(feat_patch - self.planes0_ref[i], dim=-1)
                # _, idxs = torch.topk(-L2.view(-1), 2)
                # idx = idxs[1]
                idx = torch.argmin(L2.view(-1))
                logging.info(f'move distance: {torch.norm(coordinates[idx] - point.to_tensor())}')
                if torch.allclose(coordinates[idx], point.to_tensor()):
                    logging.warning(f'Point {point} is not moving.')
                points_after_step.append(WorldPoint(coordinates[idx]))

        assert len(points_cur) == len(points_target), "Number of points should be the same."

        loss_motion = 0
        for p_cur, p_tar in zip(points_cur, points_target):
            v_cur_tar = p_tar.to_tensor() - p_cur.to_tensor()
            e_cur_tar = v_cur_tar / torch.norm(v_cur_tar)
            coordinates_cur = p_cur.gen_real_sphere_coordinates(r2, r2_step).unsqueeze(0)
            logging.info(f'Numebr of sphere coordinates: {coordinates_cur.shape[0]}, r2: {r2}, r2_step: {r2_step}')
            coordinates_step = coordinates_cur + e_cur_tar * point_step
            feat_cur = sample_from_planes(
                self.backbone_3dgan.renderer.plane_axes,
                planes,
                coordinates_cur,  # _, M, _ = coordinates.shape
                box_warp=self.backbone_3dgan.rendering_kwargs['box_warp']).detach()
            feat_step = sample_from_planes(
                self.backbone_3dgan.renderer.plane_axes,
                planes,
                coordinates_step,  # _, M, _ = coordinates.shape
                box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
            loss_motion += F.l1_loss(feat_cur, feat_step)

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
    ws = model.init_ws(z, c)
    ws = ws.detach().requires_grad_(True)
    gen_mesh_ply('output/mesh_start.ply', G, ws.detach(), mesh_res=256)
    opt = torch.optim.SGD([ws], lr=0.01)
    # eg3d/eg3d/gen_samples.py::create_samples
    points_cur = [WorldPoint(torch.tensor([0, 0, 0.2], device='cuda'))]
    points_target = [WorldPoint(torch.tensor([0.5, 0, 0.5], device='cuda'))]

    for step in range(100):
        loss, points_step, img, img_depth = model(ws=ws,
                                                  camera_parameters=c,
                                                  points_cur=points_cur,
                                                  points_target=points_target)
        logging.info(f'points_cur: {points_cur}, points_step: {points_step}')
        points_cur = points_step
        logging.info(f'loss: {loss.item()}, ws: {ws.mean()}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            save_eg3d_img(img, f'output/step_{step}.png')
            # gen_mesh_ply(f'output/mesh_{step}.ply', model.backbone_3dgan, ws.detach(), mesh_res=256)
    
    gen_mesh_ply(f'output/mesh_end.ply', model.backbone_3dgan, ws.detach(), mesh_res=256)
    print("passed")

