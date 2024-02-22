from functools import cache
import torch
from typing import Union

CUBE_LIM = 1.0

# @dataclass
class WorldPoint:
    data: torch.Tensor

    def __init__(self, data: torch.Tensor):
        assert isinstance(data,
                          torch.Tensor), f'point has to be of type torch.Tensor, got {type(data)}'
        assert data.shape == (3, ), f'point has to be of shape (3,), got {data.shape}'
        assert data.dtype == torch.float32, f'point has to be of dtype float32, got {data.dtype}'
        self.data = data

    @torch.no_grad
    def __add__(self, other: Union[torch.Tensor, 'WorldPoint']):
        if isinstance(other, torch.Tensor):
            return WorldPoint(self.data + other)
        elif isinstance(other, WorldPoint):
            return WorldPoint(self.data + other.data)
        else:
            raise RuntimeError(f'Cannot add {type(other)} to WorldPoint')

    def __repr__(self) -> str:
        x, y, z = self.data.tolist()
        cls_name = self.__class__.__name__
        return f'{cls_name}(x={x:.4f}, y={y:.4f}, z={z:.4f})'

    def to_tensor(self) -> torch.Tensor:
        return self.data

    @torch.no_grad
    def gen_sphere_coordinates(self, radius: float, step: float) -> torch.Tensor:
        coord = gen_sphere_coordinates_shift(radius, step, self.data.device) + self.data
        coord = coord.clamp(-CUBE_LIM, CUBE_LIM)
        return coord

    @torch.no_grad
    def gen_cube_coordinates(self, half_l: float, step: float) -> torch.Tensor:
        coord = gen_sphere_coordinates_shift(half_l, step, self.data.device) + self.data
        coord = coord.clamp(-CUBE_LIM, CUBE_LIM)
        return coord

@cache
@torch.no_grad
def gen_square_meshgrid(half_l, step, device):
    x_span = torch.arange(-half_l, half_l + step, step=step, device=device)
    y_span = torch.arange(-half_l, half_l + step, step=step, device=device)
    z_span = torch.arange(-half_l, half_l + step, step=step, device=device)
    return torch.meshgrid(x_span, y_span, z_span)

@cache
@torch.no_grad
def gen_sphere_coordinates_shift(radius: float, step: float, device: torch.device) -> torch.Tensor:
    xx, yy, zz = gen_square_meshgrid(radius, step, device)
    inside_sphere = xx**2 + yy**2 + zz**2 <= radius**2
    return torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
@cache
@torch.no_grad
def gen_cube_coordinates_shift(half_l: float, step: float, device: torch.device) -> torch.Tensor:
    xx, yy, zz = gen_square_meshgrid(half_l, step, device)
    return torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)

@torch.no_grad
def pixel_to_real(i: int, j: int, ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                  img_depth: torch.Tensor):
    assert ray_origin.shape == (3, )
    assert ray_dir.shape == (128, 128, 3)
    assert img_depth.shape == (128, 128)
    return WorldPoint(ray_origin + img_depth[i, j] * ray_dir[i, j])
