from dataclasses import dataclass
from training.volumetric_rendering.renderer import project_onto_planes, generate_planes
import torch
from typing import List

_PLANES = generate_planes().to('cuda')

FEAT_POINT_RES = 256

@dataclass
class FeatPoint:
    x: int
    y: int
    z: int
    resolution: int = FEAT_POINT_RES

    @torch.no_grad
    @staticmethod
    def from_list_to_tensor(points: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        return torch.stack([point.to_tensor(device) for point in points], dim=0)

    def __add__(self, other):
        return FeatPoint(self.x+other.x, self.y+other.y, self.z+other.z)

    @torch.no_grad
    def gen_real_circle_coordinates(self, radius: int, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        coordinates = torch.tensor([self.x, self.y, self.z], device=device, dtype=torch.float32).reshape(1, 1, 3)
        projected_coordinates = project_onto_planes(_PLANES, coordinates) # (3, 1, 2) xy, xz, zy
        projected_coordinates = projected_coordinates.squeeze(1) # (3, 2)
        return feat_to_real(torch.stack([
            self.gen_feat_circle_coordinate(radius, projected_coordinates[0], device=device),
            self.gen_feat_circle_coordinate(radius, projected_coordinates[1], device=device),
            self.gen_feat_circle_coordinate(radius, projected_coordinates[2], device=device)
        ], dim=0), resolution=self.resolution)

    @torch.no_grad
    def gen_feat_circle_coordinate(self, radius:int, point: torch.Tensor, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        assert point.shape == (2, )
        x, y = point.tolist()
        x_min = max(0, self.x - radius)
        x_max = min(self.resolution, self.x + radius)
        y_min = max(0, self.y - radius)
        y_max = min(self.resolution, self.y + radius)
        x_span = torch.arange(x_min, x_max)
        y_span = torch.arange(y_min, y_max)
        xx, yy = torch.meshgrid(x_span, y_span)
        inside_sphere = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere]), dim=-1).reshape(-1, 2)
        return coordinates.to(device)

    @torch.no_grad
    def gen_real_square_coordinates(self, half_l: int, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        coordinates = torch.tensor([self.x, self.y, self.z], device=device, dtype=torch.float32).reshape(1, 1, 3)
        projected_coordinates = project_onto_planes(_PLANES, coordinates) # (3, 1, 2) xy, xz, zy
        projected_coordinates = projected_coordinates.squeeze(1) # (3, 2)
        return feat_to_real(torch.stack([
            self.gen_feat_square_coordinate(half_l, projected_coordinates[0], device=device),
            self.gen_feat_square_coordinate(half_l, projected_coordinates[1], device=device),
            self.gen_feat_square_coordinate(half_l, projected_coordinates[2], device=device)
        ], dim=0), resolution=self.resolution)
    
    @torch.no_grad
    def gen_feat_square_coordinate(self, half_l: int, point: torch.Tensor, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        assert point.shape == (2, )
        x, y = point.tolist()
        x_min = max(0, self.x - half_l)
        x_max = min(self.resolution, self.x + half_l)
        y_min = max(0, self.y - half_l)
        y_max = min(self.resolution, self.y + half_l)
        x_span = torch.arange(x_min, x_max)
        y_span = torch.arange(y_min, y_max)
        xx, yy = torch.meshgrid(x_span, y_span)
        coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        return coordinates.to(device)

    @torch.no_grad
    def gen_real_sphere_coordinates(self, radius: int, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        x_min = max(0, self.x - radius)
        x_max = min(self.resolution, self.x + radius)
        y_min = max(0, self.y - radius)
        y_max = min(self.resolution, self.y + radius)
        z_min = max(0, self.z - radius)
        z_max = min(self.resolution, self.z + radius)
        x_span = torch.arange(x_min, x_max)
        y_span = torch.arange(y_min, y_max)
        z_span = torch.arange(z_min, z_max)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        inside_sphere = (xx-self.x)**2 + (yy-self.y)**2 + (zz-self.z)**2 <= radius**2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
        return feat_to_real(coordinates.to(device), resolution=self.resolution)
    
    @torch.no_grad
    def gen_real_cube_coordinates(self, half_l: int, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        x_min = max(0, self.x - half_l)
        x_max = min(self.resolution, self.x + half_l)
        y_min = max(0, self.y - half_l)
        y_max = min(self.resolution, self.y + half_l)
        z_min = max(0, self.z - half_l)
        z_max = min(self.resolution, self.z + half_l)
        x_span = torch.arange(x_min, x_max)
        y_span = torch.arange(y_min, y_max)
        z_span = torch.arange(z_min, z_max)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        cordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return feat_to_real(cordinates.to(device), resolution=self.resolution)

    @torch.no_grad
    def to_real(self, device: torch.device) -> torch.Tensor:
        return feat_to_real(torch.tensor([self.x, self.y, self.z], device=device, dtype=torch.float32), resolution=self.resolution)

    @torch.no_grad
    def from_real(self, real: torch.Tensor) -> 'FeatPoint':
        return real_to_feat(real, resolution=self.resolution)

    @torch.no_grad
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z], device=device, dtype=torch.float32)

def safe_round(x, resolution=FeatPoint.resolution):
    return max(0, min(resolution-1, round(x)))

@torch.no_grad
def tensor_to_feat(tensor: torch.Tensor, resolution: int=FeatPoint.resolution) -> FeatPoint:
    assert tensor.shape == (3,)
    return FeatPoint(safe_round(tensor[0].item(), resolution), safe_round(tensor[1].item(), resolution), safe_round(tensor[2].item(), resolution), resolution=resolution)

@torch.no_grad
def pixel_to_real(i: int, j: int, ray_origin: torch.Tensor, ray_dir: torch.Tensor, img_depth: torch.Tensor, device: torch.device='cuda'):
    assert ray_origin.shape == (3,)
    assert ray_dir.shape == (128, 128, 3)
    assert img_depth.shape == (128, 128)
    return ray_origin + img_depth[i, j] * ray_dir[i, j]

@torch.no_grad
def real_to_feat(coordinates: torch.Tensor, resolution: int) -> FeatPoint:
    assert coordinates.shape == (3,)
    p = (coordinates + 1) / 2 * resolution
    return FeatPoint(safe_round(p[0].item(), resolution), safe_round(p[1].item(), resolution), safe_round(p[2].item(), resolution), resolution=resolution)

@torch.no_grad
def feat_to_real(coordinates: torch.Tensor, resolution: int) -> torch.Tensor:
    return coordinates / resolution * 2 - 1