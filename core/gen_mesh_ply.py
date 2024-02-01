from eg3d.eg3d.shape_utils import convert_sdf_samples_to_ply

import torch
import numpy as np

# from eg3d.eg3d.gen_samples import create_samples

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 0] = -samples[:, 0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def gen_mesh_ply(
                 filename,
                 eg3d_G,
                 ws,
                 mesh_res=512,
                 max_batch=1000000,
                 device='cuda'):
    samples, voxel_origin, voxel_size = create_samples(
        N=mesh_res,
        cube_length=eg3d_G.rendering_kwargs['box_warp'] * 1)

    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1),
                         device=device)
    transformed_ray_directions_expanded = torch.zeros(
        (samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    while head < samples.shape[1]:
        torch.manual_seed(0)
        sigma = eg3d_G.sample_using_ws(
            samples[:, head:head + max_batch],
            transformed_ray_directions_expanded[:, :samples.shape[1] - head],
            ws,
            noise_mode='const')['sigma']
        sigmas[:, head:head + max_batch] = sigma
        head += max_batch

    sigmas = sigmas.reshape((mesh_res, mesh_res, mesh_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * mesh_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    convert_sdf_samples_to_ply(sigmas, voxel_origin, voxel_size, filename, level=10)
