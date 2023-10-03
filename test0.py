import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, confusion_matrix, recall_score
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score


###########################################################
###########################################################
#### RANDOM SHIT TO SAVE MODELS


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    """
    Function to save a model. 

    Args. 
        state: Dict with parameters of the model.
    """
    print("Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    """
    Function to load a model. 

    Args. 
        checkpoint: A file with the parameters of the model. 
    """
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if config['load_model'] == True:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

if config['load_model'] == True:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

def testing(model, dataloader):
    per_class_acc_pred = []
    per_class_acc_lab = []
    per_class_avg_prec_pred = []
    per_class_avg_prec_lab = []

    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device) 
        y_pred = model(inputs)

        per_class_avg_prec_pred.extend(y_pred.cpu().detach().numpy())
        per_class_avg_prec_lab.extend(torch.argmax(labels,1).cpu().detach().numpy())
        
        per_class_acc_pred.extend(torch.argmax(y_pred, 1).cpu().detach().numpy())
        per_class_acc_lab.extend(torch.argmax(labels,1).cpu().detach().numpy())

    #Accuracy per class
    tot_predics = np.array(per_class_acc_pred)
    tot_lab = np.array(per_class_acc_lab)
    matrix = confusion_matrix(tot_lab, tot_predics)

    acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
    mean_acc = np.mean(acc_per_class)
    historic_mean_acc.append(mean_acc)
    historic_perclass_acc.append(acc_per_class)


    #Average precision
    avg_prec_score = comp_average_precision(per_class_avg_prec_pred, per_class_avg_prec_lab, 6)
    mean_avg_prec = sum(avg_prec_score)/len(avg_prec_score) 
    historic_mean_avg_prec.append(mean_avg_prec)
    historic_perclass_avg_prec.append(avg_prec_score)


    #global_acc /= count
    #historic_glob_acc.append(global_acc)
    print(f"----------------- EPOCH {epoch} -----------------")
    print(f"Accuracy Per Class:\n{acc_per_class}\nmean accuracy:\n{mean_acc}\n")
    print(f"Average Precision Per Class:\n{avg_prec_score}\nmean AP:\n{mean_avg_prec}\n")

    return()