from dataclasses import dataclass
import torch

@dataclass
class Point:
    x: int
    y: int
    z: int
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z], device=device, dtype=torch.float32)
    
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'Point':
        assert tensor.shape == (3,)
        return Point(round(tensor[0].item()), round(tensor[1].item()), round(tensor[2].item()))

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y, self.z+other.z)

    def gen_sphere_coordinates(self, radius: int, device: torch.device=torch.device('cuda'), x_lim: int=1e9, y_lim: int=1e9, z_lim: int=1e9) -> torch.Tensor:
        x_min = max(0, self.x - radius)
        x_max = min(x_lim, self.x + radius)
        y_min = max(0, self.y - radius)
        y_max = min(y_lim, self.y + radius)
        z_min = max(0, self.z - radius)
        z_max = min(z_lim, self.z + radius)
        x_span = torch.arange(x_min, x_max)
        y_span = torch.arange(y_min, y_max)
        z_span = torch.arange(z_min, z_max)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        inside_sphere = (xx-self.x)**2 + (yy-self.y)**2 + (zz-self.z)**2 <= radius**2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
        return coordinates.to(device)

    def gen_cube_coordinates(self, half_l: int, device: torch.device=torch.device('cuda'), x_lim: int=1e9, y_lim: int=1e9, z_lim: int=1e9) -> torch.Tensor:
        x_min = max(0, self.x - half_l)
        x_max = min(x_lim, self.x + half_l)
        y_min = max(0, self.y - half_l)
        y_max = min(y_lim, self.y + half_l)
        z_min = max(0, self.z - half_l)
        z_max = min(z_lim, self.z + half_l)
        x_span = torch.arange(x_min, x_max)
        y_span = torch.arange(y_min, y_max)
        z_span = torch.arange(z_min, z_max)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        cordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return cordinates.to(device)

    def gen_normed_sphere_coordinates(self, radius: int,  x_lim: int, y_lim: int, z_lim: int, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        unnormed = self.gen_sphere_coordinates(radius, device=device, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)
        return unnormed / torch.tensor([x_lim, y_lim, z_lim], device=device, dtype=torch.float32) * 2 - 1

    def gen_normed_cube_coordinates(self, half_l: int, x_lim: int, y_lim: int, z_lim: int, device: torch.device=torch.device('cuda')) -> torch.Tensor:
        unnormed = self.gen_cube_coordinates(half_l, device=device, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)
        return unnormed / torch.tensor([x_lim, y_lim, z_lim], device=device, dtype=torch.float32) * 2 - 1

    @staticmethod
    def from_list_to_tensor(points: list, device: torch.device) -> torch.Tensor:
        return torch.stack([point.to_tensor(device) for point in points], dim=0)