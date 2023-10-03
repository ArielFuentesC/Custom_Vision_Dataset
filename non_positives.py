from utils import *
from params import *
from dataloaders import *
from model_structure import *
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

model=Ariel_model(6,150,[0,1,2,3,4])
model.to(device)

#per = non_positives(model,dataloader_test)
#print(per)

mn = task3(model,dataloader_val)
print(mn)
print(len(mn[0]))
print(mn[0])