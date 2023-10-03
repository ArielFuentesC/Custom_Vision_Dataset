# Dataset and data loaders;
from torch.utils.data import DataLoader, Dataset
import torch
from skimage import io
import numpy as np
from torchvision import transforms, models
from params import *

class Landscapes(torch.utils.data.Dataset):
    """
    Define a class for the Landscapes Image Classification problem.

    __getitem__. Given an index, looks for the image using its path.
    Then, transform the image into tensor and resize the image to 150x150. 
    Label is then created using an array with the same len as classes. 

    Returns: Image (tensor), label (tensor), path (string).
    """
    def __init__(self, img_path, label, transform = None):
        self.img_path = img_path
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_path[idx]
        img = io.imread(img)
        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1))
        img = img.float()

        if self.transform:
            img = self.transform(img)


        lbl = np.zeros(6)
        lbl[self.label[idx]] = 1
        lbl = torch.from_numpy(lbl)

        resize = transforms.Resize((config['size'],config['size']), antialias=True)
        img = resize(img)

        return img, lbl, self.img_path[idx]

#Defining some transforms;

transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_val = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])