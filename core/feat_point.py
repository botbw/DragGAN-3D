import torch
from typing import Union
from dataclasses import dataclass

@dataclass
class FeatPoint:
    x: int
    y: int
    z: int
    resolution: int=256

    def __repr__(self) -> str:
        x, y, z = self.x, self.y, self.z
        cls_name = self.__class__.__name__
        return f'{cls_name}(x={x}, y={y}, z={z})'

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z], device='cuda')

    @torch.no_grad
    def gen_sphere_coordinates(self, radius: int, step: int = 1) -> torch.Tensor:
        x, y, z = self.x, self.y, self.z
        x_min = max(0, x - radius)
        x_max = min(self.resolution, x + radius)
        y_min = max(0, y - radius)
        y_max = min(self.resolution, y + radius)
        z_min = max(0, z - radius)
        z_max = min(self.resolution, z + radius)
        x_span = torch.arange(x_min, x_max, step=step, device='cuda')
        y_span = torch.arange(y_min, y_max, step=step, device='cuda')
        z_span = torch.arange(z_min, z_max, step=step, device='cuda')
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
        inside_sphere = (xx - x)**2 + (yy - y)**2 + \
            (zz - z)**2 <= radius**2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
        return coordinates

    @torch.no_grad
    def gen_cube_coordinates(self, half_l: int, step: int = 1) -> torch.Tensor:
        x, y, z = self.x, self.y, self.z
        x_min = max(0, x - half_l)
        x_max = min(self.resolution, x + half_l)
        y_min = max(0, y - half_l)
        y_max = min(self.resolution, y + half_l)
        z_min = max(0, z - half_l)
        z_max = min(self.resolution, z + half_l)
        x_span = torch.arange(x_min, x_max, step=step, device='cuda')
        y_span = torch.arange(y_min, y_max, step=step, device='cuda')
        z_span = torch.arange(z_min, z_max, step=step, device='cuda')
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return coordinates


def sample(planes, coordinates):
    assert planes.shape == (1, 3, 32, 256, 256)
    yx = planes[0, 0, :, coordinates[..., 1], coordinates[..., 0]].permute(1, 0)
    zx = planes[0, 1, :, coordinates[..., 2], coordinates[..., 0]].permute(1, 0)
    yz = planes[0, 2, :, coordinates[..., 1], coordinates[..., 2]].permute(1, 0)
    return torch.stack([yx, zx, yz], dim=1).reshape(-1, 96)