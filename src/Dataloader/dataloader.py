import cv2
from torch.utils.data import Dataset, DataLoader
import os

class CarlaVeriDataset(Dataset):
    """This class inherits from the Dataset class of pytorch, hence we will define 
        The required 3 methods __init_, __len__ and __getitem__"""


    """
         Custom class for loading the Veri Dataset. 
         This class handles the infamous Veri-776 dataset having the 776 classes.
         This function takes the paths of the image as the input and converts them into useful lists like the vehcile id and the camera id.
       
    """
    
    def __init__(self, dir_path:str, transform=None)->None:
        self.dir_path=dir_path
        self.transform=transform

        # get all the image_paths
        self.image_path=glob.glob(os.path.join(self.dir_path, "*.jpg"))
        self.image_path=sorted(self.image_path)

        if self.image_path is None:
            print("There are no images in the directory.")

        self.pids=[] # This is there to store the vehicle ids
        self.camids=[] # This is there to store the camera ids
        self.valid_paths=[]

        for paths in self.image_path:
            filename=os.path.basename(paths).replace(".jpg","")
            parts=filename.split("_")

            if len(paths)>=2: ## This is a bug that we have figured it out it should not be len(paths) as this is always true
                pid=int(parts[0])
                if parts[1].startswith('c'):
                    camid=int(parts[1][1:])
                else:
                    camid=int(parts[1])

                self.pids.append(pid)
                self.camids.append(camid)
                self.valid_paths.append(paths)
        self.image_path=self.valid_paths

        """Note: The PIDS that we have here are not continuous so we have to make sure that they are continous value 
        starting from 0 to num_classes-1, Otherwise our model will collapse during training"""

        self.unique_pids=sorted(list(set(self.pids)))
        self.pid_maps={old:new for new,old in enumerate(self.unique_pids)}
        self.new_pids=[self.pid_maps[p] for p in self.pids]

        print(f"Total images: {len(self.image_path)} Total unique pids: {len(self.unique_pids)}")
        
        
    def __len__(self)-> int:
        return len(self.image_path)
    
    def __getitem__(self, index:int)-> tuple:
        """ This fnction fetches you the image and the corresponding labels"""
        path=self.image_path[index]
        camid=self.camids[index]
        pid=self.new_pids[index]
        original_pid=self.pids[index]

        image=cv2.imread(path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=Image.fromarray(image)
        if self.transform is not None:
            image=self.transform(image)

        return image, pid, camid,original_pid,path
            

        

