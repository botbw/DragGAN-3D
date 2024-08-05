from functools import cache

import torch
from deprecated import deprecated


class WorldPoint:
    tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor):
        assert isinstance(
            tensor, torch.Tensor), f'point has to be of type torch.Tensor, got {type(tensor)}'
        assert tensor.shape == (3, ), f'point has to be of shape (3,), got {tensor.shape}'
        assert tensor.dtype == torch.float32, f'point has to be of dtype float32, got {tensor.dtype}'
        self.tensor = tensor

    def __repr__(self) -> str:
        x, y, z = self.tensor.tolist()
        cls_name = self.__class__.__name__
        return f'{cls_name}(x={x:.4f}, y={y:.4f}, z={z:.4f})'


@cache
@torch.no_grad
@deprecated
def gen_cube_meshgrid(half_l: int, device: torch.device):
    span = torch.arange(-half_l, half_l + 1, device=device)
    return torch.meshgrid(span, span, span)


@cache
@torch.no_grad
def gen_square_meshgrid(half_l: int, device: torch.device):
    span = torch.arange(-half_l, half_l + 1, device=device)
    return torch.meshgrid(span, span)


@cache
@torch.no_grad
@deprecated
def gen_sphere_coordinates_shift(radius: int, device: torch.device) -> torch.Tensor:
    xx, yy, zz = gen_cube_meshgrid(radius, device)
    inside_sphere = xx**2 + yy**2 + zz**2 <= radius**2
    return torch.stack(
        (xx[inside_sphere].flatten(), yy[inside_sphere].flatten(), zz[inside_sphere].flatten()),
        dim=1)


@cache
@torch.no_grad
@deprecated
def gen_cube_coordinates_shift(half_l: int, device: torch.device) -> torch.Tensor:
    xx, yy, zz = gen_cube_meshgrid(half_l, device)
    return torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)


@cache
@torch.no_grad
def gen_circle_coordinates_shift(radius: int, device: torch.device) -> torch.Tensor:
    xx, yy = gen_square_meshgrid(radius, device)
    inside_circle = xx**2 + yy**2 <= radius**2
    xx = xx[inside_circle].flatten()
    yy = yy[inside_circle].flatten()
    all_coord = torch.concat([
        torch.stack((torch.zeros_like(xx), xx, yy), dim=-1),
        torch.stack((xx, torch.zeros_like(xx), yy), dim=-1),
        torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)
    ],
                             dim=0)
    return torch.unique(all_coord, dim=0)


@cache
@torch.no_grad
def gen_square_coordinates_shift(radius: int, device: torch.device) -> torch.Tensor:
    xx, yy = gen_square_meshgrid(radius, device)
    xx = xx.flatten()
    yy = yy.flatten()
    all_coord = torch.concat([
        torch.stack((torch.zeros_like(xx), xx, yy), dim=-1),
        torch.stack((xx, torch.zeros_like(xx), yy), dim=-1),
        torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)
    ],
                             dim=0)
    return torch.unique(all_coord, dim=0)


@torch.no_grad
def pixel_to_real(i: int, j: int, ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                  img_depth: torch.Tensor):
    assert ray_origin.shape == (3, )
    assert ray_dir.shape == (128, 128, 3)
    assert img_depth.shape == (128, 128)
    return WorldPoint(ray_origin + img_depth[i, j] * ray_dir[i, j])


if __name__ == '__main__':
    cube = gen_cube_coordinates_shift(1, torch.device('cpu'))
    square = gen_square_coordinates_shift(1, torch.device('cpu'))
    print(cube, cube.shape)
    print(square, square.shape)
