from dataclasses import dataclass
import torch

@dataclass
class Point:
    x: int
    y: int
    z: int
    
    def to_tensor(self, device):
        return torch.tensor([self.x, self.y, self.z], device=device)