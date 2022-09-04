from torch.utils.data import Dataset

import numpy as np
import os

from PIL import Image


class Images(Dataset): 
  def __init__(self, root_dir1, root_dir2, transforms=None):
    self.root_dir1 = root_dir1
    self.root_dir2 = root_dir2
    self.transforms = transforms

  def __len__(self):
    return min(len(os.listdir(self.root_dir1)), len(os.listdir(self.root_dir2))) #?????????????????????????

  def __getitem__(self, idx):
    all_names1, all_names2 = os.listdir(self.root_dir1), os.listdir(self.root_dir2)
    idx1, idx2 = np.random.randint(0, len(all_names1)), np.random.randint(0, len(all_names2))

    img_name1, img_name2 = os.path.join(self.root_dir1, all_names1[idx1]), os.path.join(self.root_dir2, all_names2[idx2])
    image1 = Image.open(img_name1).convert("RGB")
    image2 = Image.open(img_name2).convert("RGB")

    if self.transforms:
      image1 = self.transforms(image1)
      image2 = self.transforms(image2)

    return image1, image2  
