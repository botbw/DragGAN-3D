from dataclasses import dataclass
import torch

@dataclass
class Point:
    x: int
    y: int
    z: int
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z], device=device, dtype=torch.float32)
    
    def gen_sphere_coordinates(self, radius: int, device: torch.device=torch.device('cuda'), x_lim: int=1e9, y_lim: int=1e9, z_lim: int=1e9) -> torch.Tensor:
        x_min = max(0, self.x - radius)
        x_max = min(x_lim, self.x + radius + 1)
        y_min = max(0, self.y - radius)
        y_max = min(y_lim, self.y + radius + 1)
        z_min = max(0, self.z - radius)
        z_max = min(z_lim, self.z + radius + 1)
        x_span = torch.arange(x_min, x_max+1)
        y_span = torch.arange(y_min, y_max+1)
        z_span = torch.arange(z_min, z_max+1)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        inside_sphere = (xx-self.x)**2 + (yy-self.y)**2 + (zz-self.z)**2 <= radius**2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
        return coordinates.to(device)

    def gen_cube_coordinates(self, half_l: int, device: torch.device=torch.device('cuda'), x_lim: int=1e9, y_lim: int=1e9, z_lim: int=1e9) -> torch.Tensor:
        x_min = max(0, self.x - half_l)
        x_max = min(x_lim, self.x + half_l + 1)
        y_min = max(0, self.y - half_l)
        y_max = min(y_lim, self.y + half_l + 1)
        z_min = max(0, self.z - half_l)
        z_max = min(z_lim, self.z + half_l + 1)
        x_span = torch.arange(x_min, x_max+1)
        y_span = torch.arange(y_min, y_max+1)
        z_span = torch.arange(z_min, z_max+1)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        cordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return cordinates.to(device)