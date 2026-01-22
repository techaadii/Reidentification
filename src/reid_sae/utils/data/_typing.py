import torch
from typing import TypedDict
from pathlib import Path


class Veri776Sample(TypedDict):
    image: torch.Tensor
    transformed_image: torch.Tensor
    pid: int
    cam_id: int
    original_pid: int
    path: Path
