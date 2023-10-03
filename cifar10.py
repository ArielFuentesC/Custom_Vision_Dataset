# TASK 1

import os
import numpy as np
import torchvision
import torch
from torch import nn
from torchvision import transforms, models
import csv 
import pickle
import matplotlib as plt
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torch.functional as F
from torch import optim
from sklearn.metrics import confusion_matrix, average_precision_score
from utils import *
from dataloaders import *
from model_structure import *
from training import *

torch.manual_seed(config['random_state'])

#Path to global folder
datapath = "/home/bokhimi/ariel_2023/dlia/mandatory1_data"

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

train_transforms_cifar = transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

test_transforms_cifar = transforms.Compose([
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root='data/',download=True, transform = train_transforms_cifar)
test_dataset = torchvision.datasets.CIFAR10(root='data/',download=True, train=False, transform = test_transforms_cifar)

batch_size = 16
dataloader_train_cifar = DataLoader(dataset, batch_size)
dataloader_test_cifar = DataLoader(test_dataset, batch_size)


model=Ariel_model(6,224,[0,1,2,3,4])
model.to(device)
#Building the (best) model
fit(model, dataloader_train_cifar, dataloader_test_cifar, epochs = config['num_epochs'])
    
mn = task3(model,dataloader_test_cifar)
print(mn)
print(len(mn[0]))
print(mn[0])

