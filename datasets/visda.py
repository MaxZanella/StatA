import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional


class VisdaTest(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        self.root = root
        self.transforms = transforms
        self.img_list = np.loadtxt(root + 'image_list.txt', dtype=str)

    def __len__(self):
        return self.img_list.shape[0]

    def __getitem__(self, idx):
        name = self.img_list[idx][0]
        label = int(self.img_list[idx][1])

        img = Image.open(self.root + 'test/' + name)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label