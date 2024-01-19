from dataclasses import dataclass
import torch

@dataclass
class Point:
    x: int
    y: int
    z: int

def get_feature_from_planes(p: Point, planes: torch.Tensor):
    assert planes.shape[0] == 3, "three planes"
    