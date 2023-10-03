#Useful functions. 

import os 
import torch
from torch import nn
from torchvision import transforms, models
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, average_precision_score
from matplotlib import pyplot as plt
import numpy as np
import pickle
from utils import *
from params import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataloading(datapath, classes):
    """
    Function to load the dataset. 
    Args.
        datapath: Directory to the general folder which contains the subfolders that 
        define the dataset classes.
        classes: List of the classes of the dataset. 
    Returns.
        imgs. List of all the paths to the images in the dataset.
        labels. List of all the labels of the dataset.
    """
    imgs = []
    labels = []

    #Iterating over the subdirectories to get the list of paths and labels
    for i, clss in enumerate(classes):
        files = sorted(os.listdir(os.path.join(datapath, clss)))

        paths = [os.path.join(datapath, clss, fn) for fn in files]

        size = len(paths)

        imgs.extend(paths)
        labels.extend([i] * size)
    return imgs, labels

def splitting(imgs, labels,val_test_size=5000, test_size=3000):
    """
    Splits the dataset in three sets; training, validation and test. 
    The split is stratified over labels and uses a fixed random_state. 
    Args. 
        imgs. List of paths to all the images in the dataset.
        labels. List of labels of all the images in the dataset.
        val_test_size. Size of the validation plus the test set.
        test_size. Desired test size. 
    Returns. 
        imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test.
        Lists of the labels and paths to the images corresponding to each subset. 
    """
    dataset_size = len(imgs)

    test_val_percentage = val_test_size/dataset_size

    imgs_train, imgs_val_test, labels_train, labels_val_test = train_test_split(imgs, labels, test_size=test_val_percentage,
                                                                                random_state=config['random_state'], stratify=labels)
    tst_sz = test_size/val_test_size
    imgs_val, imgs_test, labels_val, labels_test = train_test_split(imgs_val_test, labels_val_test, test_size=tst_sz,
                                                                    random_state=config['random_state'], stratify=labels_val_test)

    print(f"Validation size: {len(imgs_val)}\nTest size: {len(imgs_test)}\nTrain_size {len(imgs_train)}")

    return imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test

def disjointness(train, val, test):
    """
    Verify disjointness of the train, val and test sets through intersections between sets. 
    Args.
        train. List of all the images corresponding to the training set.
        val. List of all the images corresponding to the validation set.
        test. List of all the images corresponding to the test set.
    """
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    int_train_val = train_set.intersection(val_set) == set()
    int_train_test = train_set.intersection(test_set) == set()
    int_val_test = val_set.intersection(test_set) == set()

    if int_train_val and int_train_test and int_val_test == True:
        print(f"The intersections one-to-one among the subsets are equal to the empty set. Therefore, they're disjoint.")
    else:
        print(f"The subsets are not disjoint.")

def comp_average_precision(scores, y_label, num_clss):
    """
    Compute Average Precision for each class. 

    Args.
        scores: softmax values of predictions. 
        y_label: ground truth labels.
        num_clss: number of classes.
    
    Returns. 
        avg_prec_glob: Array with the average precision score for each class.
    """
    avg_prec_glob = []

    for clss in range(num_clss): #For each class

        #Binarizing
        scores_bin = [x[clss] for x in scores]
        y_lab_bin = np.zeros(len(y_label))

        for i, lab in enumerate(y_label):
            if lab == clss:
                y_lab_bin[i] = 1 

        prec_score = average_precision_score(y_lab_bin,scores_bin)
        avg_prec_glob.append(prec_score)
    
    return avg_prec_glob

def top10(probs, paths, classes=[0,3,5]):
    """
    Given the desired classes, it returns the top 10 and bottom 10 images
    based on the score obtained. 
    Args. 
        probs. Softmax scores.
        paths. Paths to the images.
        classes. Classes of interest. 
    Returns. 
        tops_tot. List of tuples (score, path to the image) for the top 10 images.
        bot_tot. List of tuples (score, path to the image) for the bot 10 images.
    ***I couldn't plot on the server, so via the config dic saved a pickle to plot them locally.
    """
    tops_tot = []
    bot_tot = []
    for cls in classes:
        tops = []
        bots = []
        
        per_class = [column[cls] for column in probs]
        bot10 = np.argsort(per_class)[:10]
        top10 = np.argsort(per_class)[-10:]
        tops.append([(per_class[i], paths[i]) for i in top10])
        bots.append([(per_class[i],paths[i]) for i in bot10])
        tops_tot.append(tops)
        bot_tot.append(bots)
    return tops_tot, bot_tot

def non_positives(model,data_loader,hks=200,batch_size=config['batch_size']):
    """
    Computes the percentage of non positives values over certain layers. 
    Args.
        model. Pytorch model.
        data_loader. Py torch dataloader.
        hks. Number of desired hooks. 
        batch_size. Batch size
    """
    i = 0
    iter = hks//batch_size #Calculate the number of required batches 
    dict_hooks = {} #Dict for features
    for inputs, labels, paths in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_pred, new_hooks = model(inputs)
        for layer in new_hooks.keys(): #For each layer
            if i == 0:
                dict_hooks[layer] = new_hooks[layer]
            if i <= iter and i>1:
                dict_hooks[layer] = torch.cat((dict_hooks[layer],new_hooks[layer])) #Saving the tensors
                i += 1
            if i> iter and dict_hooks[layer].shape[0] != hks:
                dict_hooks[layer] = torch.cat((dict_hooks[layer],new_hooks[layer][:hks//batch_size])) #Completing the desired number of hooks
    percentage_layers = []
    for layer in new_hooks.keys():
        non_positive = dict_hooks[layer] <= 0
        per = non_positive.float().sum()/np.prod(list(dict_hooks[layer].size())) #Computing the number of non positives values
        percentage_layers.append(per.item())
    print(new_hooks.keys())
    return percentage_layers

def task3(model,data_loader,hks=200,batch_size=config['batch_size']):
    """
    Computes the average of a feature map over all spatial dimensions. 
    Args.
        model. Pytorch model.
        data_loader. Py torch dataloader.
        hks. Number of desired hooks. 
        batch_size. Batch size
    """
    i = 0
    iter = hks//batch_size #Calculate the number of required batches 
    dict_hooks = {} #Dict for features
    for inputs, labels, paths in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_pred, new_hooks = model(inputs)
        for layer in new_hooks.keys(): #For each layer
            if i == 0:
                dict_hooks[layer] = new_hooks[layer]
            if i <= iter and i>1:
                dict_hooks[layer] = torch.cat((dict_hooks[layer],new_hooks[layer])) #Saving the tensors
                i += 1
            if i> iter and dict_hooks[layer].shape[0] != hks:
                dict_hooks[layer] = torch.cat((dict_hooks[layer],new_hooks[layer][:hks//batch_size])) #Completing the desired number of hooks
    mean_layers = []
    for layer in new_hooks.keys():
        mn = torch.mean(dict_hooks[layer],(2, 3)) #Mean over the spatial dimensions 
        cov_mat = torch.cov(torch.transpose(mn, 0,1))
        eig_val, eig_vec = torch.linalg.eig(cov_mat)
        abs = torch.absolute(eig_val)
        sortedabs, idx = torch.sort(abs)
        mean_layers.append(sortedabs.tolist())

    if config['eig_val_graph'] == True:
        file = open("cifar10.pkl", "wb")
        pickle.dump(mean_layers, file)

    return mean_layers