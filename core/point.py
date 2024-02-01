from training.volumetric_rendering.renderer import project_onto_planes, generate_planes
import torch
from typing import Union

_PLANES = generate_planes().to('cuda')  # TODO @botbw: remove this

CUBE_LIM = 1.0

# @dataclass
class WorldPoint:
    data: torch.Tensor

    @property
    def x(self) -> float:
        return self.data[0].item()

    @property
    def y(self) -> float:
        return self.data[1].item()

    @property
    def z(self) -> float:
        return self.data[2].item()

    def __init__(self, point: torch.Tensor):
        assert isinstance(point,
                          torch.Tensor), f'point has to be of type torch.Tensor, got {type(point)}'
        assert point.shape == (3, ), f'point has to be of shape (3,), got {point.shape}'
        assert point.dtype == torch.float32, f'point has to be of dtype float32, got {point.dtype}'
        self.data = point

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
        return f'{cls_name}(x={x}, y={y}, z={z})'

    def to_tensor(self) -> torch.Tensor:
        return self.data

    @torch.no_grad
    def gen_circle_coordinates(self, radius: float, step: float) -> torch.Tensor:
        coordinates = self.data.reshape(1, 1, 3)
        projected_coordinates = project_onto_planes(_PLANES, coordinates)  # (3, 1, 2) xy, xz, zy
        projected_coordinates = projected_coordinates.squeeze(1)  # (3, 2)
        device = coordinates.device
        return torch.stack([
            self.__gen_circle_coordinate(radius, projected_coordinates[0], device=device),
            self.__gen_circle_coordinate(radius, projected_coordinates[1], device=device),
            self.__gen_circle_coordinate(radius, projected_coordinates[2], device=device)
        ],
                           dim=0)

    @torch.no_grad
    def __gen_circle_coordinate(self, radius: float, step: float, point: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
        assert point.shape == (2, )
        x, y = point.tolist()
        x_min = max(-CUBE_LIM, self.x - radius)
        x_max = min(CUBE_LIM, self.x + radius)
        y_min = max(-CUBE_LIM, self.y - radius)
        y_max = min(CUBE_LIM, self.y + radius)
        x_span = torch.arange(x_min, x_max, step=step, device=device)
        y_span = torch.arange(y_min, y_max, step=step, device=device)
        xx, yy = torch.meshgrid(x_span, y_span)
        inside_sphere = (xx - x)**2 + (yy - y)**2 <= radius**2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere]), dim=-1).reshape(-1, 2)
        return coordinates

    @torch.no_grad
    def gen_square_coordinates(self, half_l: float, step: float) -> torch.Tensor:
        coordinates = self.data.reshape(1, 1, 3)
        projected_coordinates = project_onto_planes(_PLANES, coordinates)  # (3, 1, 2) xy, xz, zy
        projected_coordinates = projected_coordinates.squeeze(1)  # (3, 2)
        device = coordinates.device
        return torch.stack([
            self.__gen_square_coordinate(half_l, projected_coordinates[0], device=device),
            self.__gen_square_coordinate(half_l, projected_coordinates[1], device=device),
            self.__gen_square_coordinate(half_l, projected_coordinates[2], device=device)
        ],
                           dim=0)

    @torch.no_grad
    def __gen_square_coordinate(
        self,
        half_l: float,
        step: float,
        point: torch.Tensor,
        device: torch.device = torch.device('cuda')) -> torch.Tensor:
        assert point.shape == (2, )
        x, y = point.tolist()
        x_min = max(-CUBE_LIM, self.x - half_l)
        x_max = min(CUBE_LIM, self.x + half_l)
        y_min = max(-CUBE_LIM, self.y - half_l)
        y_max = min(CUBE_LIM, self.y + half_l)
        x_span = torch.arange(x_min, x_max, step=step, device=device)
        y_span = torch.arange(y_min, y_max, step=step, device=device)
        xx, yy = torch.meshgrid(x_span, y_span)
        coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        return coordinates

    @torch.no_grad
    def gen_sphere_coordinates(self, radius: float, step: float) -> torch.Tensor:
        x_min = max(-CUBE_LIM, self.x - radius)
        x_max = min(CUBE_LIM, self.x + radius)
        y_min = max(-CUBE_LIM, self.y - radius)
        y_max = min(CUBE_LIM, self.y + radius)
        z_min = max(-CUBE_LIM, self.z - radius)
        z_max = min(CUBE_LIM, self.z + radius)
        x_span = torch.arange(x_min, x_max, step=step, device=self.data.device)
        y_span = torch.arange(y_min, y_max, step=step, device=self.data.device)
        z_span = torch.arange(z_min, z_max, step=step, device=self.data.device)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
        inside_sphere = (xx - self.x)**2 + (yy - self.y)**2 + (zz - self.z)**2 <= radius**2
        coordinates = torch.stack((xx[inside_sphere], yy[inside_sphere], zz[inside_sphere]), dim=1)
        return coordinates

    @torch.no_grad
    def gen_cube_coordinates(self, half_l: float, step: float) -> torch.Tensor:
        x_min = max(-CUBE_LIM, self.x - half_l)
        x_max = min(CUBE_LIM, self.x + half_l)
        y_min = max(-CUBE_LIM, self.y - half_l)
        y_max = min(CUBE_LIM, self.y + half_l)
        z_min = max(-CUBE_LIM, self.z - half_l)
        z_max = min(CUBE_LIM, self.z + half_l)
        x_span = torch.arange(x_min, x_max, step=step, device=self.data.device)
        y_span = torch.arange(y_min, y_max, step=step, device=self.data.device)
        z_span = torch.arange(z_min, z_max, step=step, device=self.data.device)
        xx, yy, zz = torch.meshgrid(x_span, y_span, z_span)
        coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return coordinates


@torch.no_grad
def pixel_to_real(i: int, j: int, ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                  img_depth: torch.Tensor):
    assert ray_origin.shape == (3, )
    assert ray_dir.shape == (128, 128, 3)
    assert img_depth.shape == (128, 128)
    return WorldPoint(ray_origin + img_depth[i, j] * ray_dir[i, j])
