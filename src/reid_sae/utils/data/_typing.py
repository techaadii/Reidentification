import torch
from typing import TypedDict


class Veri776Sample(TypedDict):
    image: torch.Tensor
    pid: int
    cam_id: int
    original_pid: int
