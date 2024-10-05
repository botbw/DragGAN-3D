import logging
import math
import os
from copy import deepcopy
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.gen_mesh_ply import gen_mesh_ply
from core.log import setup_logger
from core.utils import seed_everything, save_eg3d_img
from core.world_point import WorldPoint, r1_coordinates_shift, r2_coordinates_shift
from eg3d.eg3d.training.triplane import TriPlaneGenerator
from training.volumetric_rendering import math_utils
from training.volumetric_rendering.renderer import sample_from_planes

import matplotlib.pyplot as plt

# yapf: disable
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
        return self.renderer.run_model(planes, self.decoder, coordinates, directions,
                                       self.rendering_kwargs)

    backbone.synthesis = partial(backbone_synthesis, backbone)
    backbone.forward = partial(backbone_forward, backbone)
    backbone.renderer.forward = partial(renderer_forward, backbone.renderer)
    backbone.sample_using_ws = partial(sample_using_ws, backbone)

    return backbone
# yapf: enable


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
        self.camera_parameters = camera_parameters
        self.planes0 = synthesised['planes'].detach()
        self.planes_resolution = synthesised['planes'].shape[-1]
        self.init_run = True
        return ws

    def forward(
        self,
        ws: torch.Tensor,
        points_cur: List[WorldPoint],
        points_target: List[WorldPoint],
        r1: float,
        r2: float,
        pixel_step: float,
        mask_loss_lambda: float,
    ):
        assert self.init_run, "Please call init_ws() first to get ws_0"
        assert len(points_cur) == len(points_target), "Number of points should be the same."

        synthesised = self.backbone_3dgan.synthesis(ws, self.camera_parameters)
        img, planes = synthesised['image'], synthesised['planes']

        if self.planes0_points_ref is None:
            points_feat = []
            for p_cur in points_cur:
                feat = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    self.planes0,
                    p_cur.tensor.unsqueeze(0).unsqueeze(0),
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
                points_feat.append(feat.reshape(96))
            self.planes0_points_ref = torch.stack(points_feat, dim=0)

        # Point tracking with feature matching
        points_after_step = []

        r1_in_pixel = r1 * self.planes_resolution
        with torch.no_grad():
            for i, (p_cur, p_tar) in enumerate(zip(points_cur, points_target)):
                coordinates = p_cur.tensor + pixel_step * r1_coordinates_shift(
                    r1_in_pixel, self.device)
                feat_cur = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes,
                    coordinates.unsqueeze(0),
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp']).permute(0, 2, 1,
                                                                                       3).view(
                                                                                           -1, 96)
                dis = (feat_cur - self.planes0_points_ref[i]).norm(dim=-1)
                idx = torch.argmin(dis)
                move_vector = (coordinates[idx].float() - p_cur.tensor)
                points_after_step.append(WorldPoint(coordinates[idx]))
                logging.info(
                    f'Point track:\n\t {r1_in_pixel=} {self.planes_resolution=} numebr of cube coordinates: {coordinates.shape[0]}\n\tmove vector: {move_vector}\n\tdistance: {move_vector.float().norm()}'
                )
                if torch.allclose(coordinates[idx], p_cur.tensor):
                    logging.warning(f'Point track: point {p_cur} is not moving.')

        loss = 0

        loss_motion = 0
        r2_in_pixel = r2 * self.planes_resolution
        for p_cur, p_tar in zip(points_cur, points_target):
            with torch.no_grad():
                v_cur_tar = p_tar.tensor - p_cur.tensor
                e_cur_tar = v_cur_tar / (torch.norm(v_cur_tar.float()) + 1e-20)
                coordinates_cur = p_cur.tensor + pixel_step * r2_coordinates_shift(
                    r2_in_pixel, self.device)
                coordinates_step = coordinates_cur + pixel_step * e_cur_tar
                feat_cur = sample_from_planes(
                    self.backbone_3dgan.renderer.plane_axes,
                    planes,
                    coordinates_cur.unsqueeze(0),
                    box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
            feat_step = sample_from_planes(
                self.backbone_3dgan.renderer.plane_axes,
                planes,
                coordinates_step.unsqueeze(0),
                box_warp=self.backbone_3dgan.rendering_kwargs['box_warp'])
            loss_motion += F.l1_loss(feat_cur.detach(), feat_step)
            logging.info(
                f'{r2_in_pixel=} \n\tMotion supervision: numebr of sphere coordinates: {coordinates_cur.shape[0]}')

        loss += loss_motion

        mask_loss = 0
        for p_cur, p_tar in zip(points_cur, points_target):
            with torch.no_grad():
                mask = torch.ones(3,
                            self.planes_resolution,
                            self.planes_resolution,
                            device=self.device)
                p_mid_tensor = (0.5 * (p_cur.tensor.float() + p_tar.tensor))
                R_real = max(r2, (p_tar.tensor - p_cur.tensor).norm() / 2)
                R_pixel = R_real * self.planes_resolution
                coordinates = p_mid_tensor + pixel_step * r2_coordinates_shift(
                    R_pixel, self.device)
                coordinates = (((
                    (2 / self.backbone_3dgan.rendering_kwargs['box_warp']) * coordinates) + 1) /
                               2 * self.planes_resolution).round().long().clamp_max(255)
                mask[0, coordinates[..., 1], coordinates[..., 0]] = 0
                mask[1, coordinates[..., 2], coordinates[..., 0]] = 0
                mask[2, coordinates[..., 1], coordinates[..., 2]] = 0
            # mask uneditable area
            logging.info(f"mask percent {mask.sum() / mask.numel()}")
            mask_loss += mask_loss_lambda * F.l1_loss(
                planes.permute(0, 2, 1, 3, 4) * mask,
                self.planes0.permute(0, 2, 1, 3, 4) * mask)

        loss += mask_loss

        logging.info(f'loss: {loss}, motion_loss: {loss_motion} mask_loss: {mask_loss}')
        logging.debug(f"{r1_in_pixel=} {r2_in_pixel=} {pixel_step=}")

        return loss, points_after_step, img, synthesised['depth_image']


RANDOM = os.environ.get('RANDOM', '0').lower() in ['1', 'true']
LATENT_PATH = os.environ.get('LATENT_PATH', 'examples/latents/biden')
CKPTS_PATH = os.environ.get('CKPTS_PATH', 'ckpts/afhqcats512-128.pkl')
if __name__ == "__main__":
    import pickle

    from eg3d.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
    seed = 10086
    device = torch.device('cuda')
    seed_everything(seed)
    setup_logger(logging.DEBUG)

    ckpt = CKPTS_PATH
    with open(ckpt, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    model = DragStep(wrap_eg3d_backbone(G), device=device)
    z = torch.randn([1, G.z_dim]).cuda()

    # see C_xyz at eg3d/docs/camera_coordinate_conventions.jpg
    cam2world_pose = LookAtPoseSampler.sample(horizontal_mean=0.5 * math.pi,
                                                vertical_mean=0.5 * math.pi,
                                                lookat_position=torch.tensor(
                                                    G.rendering_kwargs['avg_camera_pivot'],
                                                    device=device),
                                                radius=G.rendering_kwargs['avg_camera_radius'],
                                                device=device)

    fov_deg = 18.837  # 0 -> inf : zoom-in -> zoom-out
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16),
                    intrinsics.reshape(-1, 9)], 1)  # camera parameters

    if RANDOM:
        ws0 = model.init_ws(z, c).detach()
    else:
        try:
            cam_dict = torch.load(LATENT_PATH + '/cam_params.pt')
            c = torch.cat(
            [cam_dict['cam_params'].reshape(-1, 16), cam_dict['cam_intrinsics'].reshape(-1, 9)],
            1).cuda()
        except Exception:
            pass
        model.init_ws(z, c)
        ws0 = torch.load(LATENT_PATH + '/latents.pt')['w_plus'].cuda()

    ws = ws0.detach().clone().requires_grad_(True)
    print(f'ws0: {ws0.shape}')

    if not os.path.exists('examples/outputs'):
        os.makedirs('examples/outputs')

    os.system('rm examples/outputs/*')
    gen_mesh_ply(f'examples/outputs/{seed}_start.ply', G, ws.detach(), mesh_res=256)
    lr = 1e-3
    opt = torch.optim.SGD([ws], lr=lr)
# [13:34:19] [Picked]	- Tri#376952 
    points_ori = [
        WorldPoint(torch.tensor((-0.255067,0.243902,-0.041737), device=device)),
        WorldPoint(torch.tensor((0.234031,0.250351,-0.063874), device=device)),
    ]

    points_target = [
        WorldPoint(torch.tensor((-0.15,0.243902,-0.041737), device=device)),
        WorldPoint(torch.tensor((0.15,0.250351,-0.063874), device=device)),
    ]

    points_cur = deepcopy(points_ori)

    l_w = 14  # fisrt l_w layers
    r1 = 3 / model.planes_resolution
    r2 = 12 / model.planes_resolution
    pixel_step: float = 0.05 / model.planes_resolution
    mask_loss_lambda: float = 5.0

    print(f"{r1=} {r2=} {pixel_step=} {mask_loss_lambda=} {model.planes_resolution=} {G.z_dim=}")
    torch.save({'w_plus': ws}, f'examples/outputs/latent_{seed}_start.pt')
    try:
        for step in range(1):
            assert torch.allclose(ws[:, l_w:, :], ws0[:, l_w:, :])
            assert not (step != 0 and l_w != 0 and torch.allclose(ws[:, :l_w, :], ws0[:, :l_w, :]))
            ws_input = torch.cat([ws[:, :l_w, :], ws0[:, l_w:, :]], dim=1)
            loss, points_step, img, img_depth = model(ws=ws_input,
                                                      points_cur=points_cur,
                                                      points_target=points_target,
                                                      r1=r1,
                                                      r2=r2,
                                                      pixel_step=pixel_step,
                                                      mask_loss_lambda=mask_loss_lambda)
            logging.info(
                f'step: {step}\npoints_ori: {points_ori}\npoints_cur: {points_cur}\npoints_tar: {points_target}'
            )
            points_cur = points_step
            assert not loss.isnan()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step == 0 or (step + 1) % 100 == 0:
                save_eg3d_img(img, f'examples/outputs/step_{step + 1}.png')

            if not any((p_cur.tensor - p_tar.tensor).norm() > 0.001 for p_cur, p_tar in zip(points_cur, points_target)):
                logging.info("Early stop since all points move to corresponding target")
                break

    except KeyboardInterrupt:
        pass

    logging.info(f'points_end: {points_cur}')
    save_eg3d_img(img, f'examples/outputs/{seed}_end.png')
    torch.save({'w_plus': ws}, f'examples/outputs/latent_{seed}_end.pt')
    gen_mesh_ply(f'examples/outputs/{seed}_end.ply',
                 model.backbone_3dgan,
                 ws.detach(),
                 mesh_res=256)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5), dpi=70)

    synthesised = G.synthesis(ws0, c)
    image = synthesised['image']
    image = image[0].detach()
    image = (image + 1) / 2
    image = image.permute(1, 2, 0).cpu().numpy()
    axs[0].imshow(image)
    axs[0].axis('off')

    for i, horrizontal_mean in enumerate([0.35 * math.pi, 0.5 * math.pi, 0.65 * math.pi]):
        cam2world_pose = LookAtPoseSampler.sample(horizontal_mean=horrizontal_mean,
                                                vertical_mean=0.5 * math.pi,
                                                lookat_position=torch.tensor(
                                                    G.rendering_kwargs['avg_camera_pivot'],
                                                    device=device),
                                                radius=G.rendering_kwargs['avg_camera_radius'],
                                                device=device)

        fov_deg = 18.837  # 0 -> inf : zoom-in -> zoom-out
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16),
                        intrinsics.reshape(-1, 9)], 1)  # camera parameters
        synthesised = G.synthesis(ws, c)
        image = synthesised['image']
        image = image[0].detach()
        image = (image + 1) / 2
        image = image.permute(1, 2, 0).cpu().numpy()
        axs[i + 1].imshow(image)
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('examples/outputs/quadruple_view.png', bbox_inches='tight', pad_inches=0.1)

    print("Finished")

