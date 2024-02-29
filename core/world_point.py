from functools import cache
import torch
from typing import Union

# @dataclass
class WorldPoint:
    data: torch.Tensor

    def __init__(self, data: torch.Tensor):
        assert isinstance(data,
                          torch.Tensor), f'point has to be of type torch.Tensor, got {type(data)}'
        assert data.shape == (3, ), f'point has to be of shape (3,), got {data.shape}'
        assert data.dtype == torch.float32, f'point has to be of dtype float32, got {data.dtype}'
        self.data = data

    def __repr__(self) -> str:
        x, y, z = self.data.tolist()
        cls_name = self.__class__.__name__
        return f'{cls_name}(x={x:.4f}, y={y:.4f}, z={z:.4f})'

    @torch.no_grad
    def gen_sphere_coordinates(self, radius, point_step) -> torch.Tensor:
        coord = gen_sphere_coordinates_shift(radius, self.data.device) * point_step + self.data
        return coord

    @torch.no_grad
    def gen_cube_coordinates(self, half_l, point_step) -> torch.Tensor:
        coord = gen_sphere_coordinates_shift(half_l, self.data.device) * point_step + self.data
        return coord

@cache
@torch.no_grad
def gen_square_meshgrid(half_l, device):
    span = torch.arange(-half_l, half_l + 1, device=device)
    return torch.meshgrid(span, span, span)

@cache
@torch.no_grad
def gen_sphere_coordinates_shift(radius, device: torch.device) -> torch.Tensor:
    xx, yy, zz = gen_square_meshgrid(radius, device)
    inside_sphere = xx**2 + yy**2 + zz**2 <= radius**2
    return torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
@cache
@torch.no_grad
def gen_cube_coordinates_shift(half_l, device: torch.device) -> torch.Tensor:
    xx, yy, zz = gen_square_meshgrid(half_l, device)
    return torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)

@torch.no_grad
def pixel_to_real(i: int, j: int, ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                  img_depth: torch.Tensor):
    assert ray_origin.shape == (3, )
    assert ray_dir.shape == (128, 128, 3)
    assert img_depth.shape == (128, 128)
    return WorldPoint(ray_origin + img_depth[i, j] * ray_dir[i, j])
