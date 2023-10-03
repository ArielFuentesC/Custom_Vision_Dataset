# TASK 1

import os
import numpy as np
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


print(f"Current device: {device}")


imgs, labels = dataloading(datapath, classes)

imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test = splitting(imgs, labels)



#Creating Datasets for the three sets;
train_data = Landscapes(imgs_train, labels_train, transforms_train)
val_data = Landscapes(imgs_val, labels_val, transforms_val)
test_data = Landscapes(imgs_test, labels_test)

#Dataloaders
dataloader_train = DataLoader(train_data, batch_size=config["batch_size"], num_workers=config["num_workers"])
dataloader_val = DataLoader(val_data, batch_size=config["batch_size"], num_workers=config["num_workers"])
dataloader_test = DataLoader(test_data, batch_size=config["batch_size"], num_workers=config["num_workers"])

model=Ariel_model(6,config['size'],[0,1,2,3,4])
model.to(device)
#Building the (best) model
if config['training_model'] == True:
    fit(model, dataloader_train, dataloader_val, epochs = config['num_epochs'])
    
if config['loading_model'] == True:
    model.load_state_dict(torch.load('model_saved.pt'))

dict_pred_lab = evaluate(model,dataloader_test)

