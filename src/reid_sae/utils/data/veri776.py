import os
from glob import glob
from pathlib import Path

import torch
from torchvision.io.image import read_image
from torchvision.transforms import v2

from reid_sae.utils.data._typing import Veri776Sample


class CarlaVeriDataset(torch.utils.data.Dataset[Veri776Sample]):
    """
    This class inherits from the torch Dataset class and helps creating the batches of dataset as per requirement
    
    This is the custom class to load the Vehcile Reidentification dataset.\
    This class handles the infamous vehicle reidentification dataset called the Veri- 776
    TODO : Explain the input and output, add the attributes of the class
    """

    def __init__(self, dir_path: str, transform: v2.Compose) -> None:
        self.image_dir: str = dir_path
        self.transform: v2.Compose = transform

        self.image_paths = glob(os.path.join(self.image_dir, "*.jpg"))
        self.image_paths = sorted(self.image_paths)

        if self.image_paths is None:
            print("There are no images in the directory")

        self.pids: list = []
        self.camids: list = []
        self.valid_paths: list = []

        for image_path in self.image_paths:
            file_name: str = os.path.basename(image_path).replace(".jpg", "")
            parts = file_name.split("_")

            if len(parts) >= 2:
                pid: int = int(parts[0])
                if parts[1].startswith("c"):
                    camid: int = int(parts[1][1:])
                else:
                    camid: int = int(parts[1])

                self.pids.append(pid)
                self.camids.append(camid)
                self.valid_paths.append(image_path)

        self.image_path = self.valid_paths

        """NOTE: 
            As you can see in the dataset that the first part of parts is the vehicle ID and the second part of parts is the camera id starting with C.
            Since it is starting with C so the if else that we have written above, makes sure that if the part 2 starts with C then we skip that C and take the values after that and if it doesn't start with C then we take the value of the part2 directly
        """
        self.unique_pids: list = sorted(list(set(self.pids)))
        self.pid_maps: dict = {old: new for new, old in enumerate(self.unique_pids)}
        self.new_pids: list = [self.pid_maps[p] for p in self.pids]

        print(
            f"Total_ images: {len(self.image_path)}, Total unique pids : {len(self.unique_pids)}"
        )

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, index) -> Veri776Sample:
        path: str = self.image_path[index]
        camid: int = self.camids[index]
        pid: int = self.new_pids[index]
        original_pid: int = self.pids[index]

        image: torch.Tensor = read_image(path=path)
        transformed_image: torch.Tensor = self.transform(image)

        return Veri776Sample(
            image=image,
            transformed_image=transformed_image,
            pid=pid,
            cam_id=camid,
            original_pid=original_pid,
            path=Path(path),
        )
