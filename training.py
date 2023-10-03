from params import *
from utils import *
import torch
from torch import optim
from torch import nn
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit(model, dataloader_train, dataloader_val, epochs=config['num_epochs']):
    """
    Computes the training of a model. 
    Args.
        model. Pytorch model.
        dataloader_train. Dataloader corresponding to the train set.
        dataloader_val. Dataloader corresponding to the validation set.
        epochs. Number of epochs.
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    
    historic_mean_acc = []
    historic_mean_avg_prec = []
    historic_perclass_acc = []
    historic_perclass_avg_prec = []
    loss_epoch_train = []
    loss_epoch_val = []

    #Training loop;
    for epoch in range(epochs):
        model.train()
        loss_batch = []
        for inputs, labels, paths in dataloader_train:
            
            inputs = inputs.to(device)
            labels = labels.to(device) 
            y_pred, hooks = model(inputs)

            loss = loss_fn(y_pred, labels)
            loss_batch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        scheduler.step()
        loss_batch = np.array(loss_batch)
        loss_epoch_train.append(np.mean(loss_batch))
        
        per_class_acc_pred = []
        per_class_acc_lab = []
        per_class_avg_prec_pred = []
        per_class_avg_prec_lab = []

        loss_batch = []
        glob_paths = []
        model.eval()
        with torch.no_grad():
            
            for inputs, labels, paths in dataloader_val:
                inputs = inputs.to(device)
                labels = labels.to(device) 
                y_pred, _ = model(inputs)
                loss = loss_fn(y_pred, labels)
                loss_batch.append(loss.item())

                per_class_avg_prec_pred.extend(y_pred.cpu().detach().numpy())
                per_class_avg_prec_lab.extend(torch.argmax(labels,1).cpu().detach().numpy())
                
                per_class_acc_pred.extend(torch.argmax(y_pred, 1).cpu().detach().numpy())
                per_class_acc_lab.extend(torch.argmax(labels,1).cpu().detach().numpy())

                if config['top10'] == True:
                    glob_paths.extend(paths)  

            loss_batch = np.array(loss_batch)
            loss_epoch_val.append(np.mean(loss_batch))
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

        if config['saving_model'] == True:
            torch.save(model.state_dict(), 'model_saved.pt')
        
        if config['graphs'] == True:
            file = open("best_model.pkl", "wb")
            pickle.dump(historic_mean_acc, file)
            pickle.dump(historic_mean_avg_prec, file)
            pickle.dump(historic_perclass_acc, file)
            pickle.dump(historic_perclass_avg_prec, file)
            pickle.dump(loss_epoch_train, file)
            pickle.dump(loss_epoch_val, file)
        
        if config['top10'] == True and epoch == config['num_epochs']-1:
            print("Saving images")
            tops, bots = top10(per_class_avg_prec_pred,glob_paths)
            print(tops)
            file = open("top_bot.pkl", "wb")
            pickle.dump(tops, file)
            pickle.dump(bots, file)         

        print(f"----------------- EPOCH {epoch} -----------------")
        print(f"Accuracy Per Class:\n{acc_per_class}\nmean accuracy:\n{mean_acc}\n")
        print(f"Average Precision Per Class:\n{avg_prec_score}\nmean AP:\n{mean_avg_prec}\n")
        print(f"Loss epoch {epoch}. Train: {loss_epoch_train[-1]}. Val: {loss_epoch_val[-1]}")

def evaluate(model, dataloader):
    """
    Function to evaluate the model. 
    Args.
        model. Pytorch model.  
        dataloader. Data to evaluate. 
    """

    model.to(device)
    per_class_acc_pred = []
    per_class_acc_lab = []
    per_class_avg_prec_pred = []
    per_class_avg_prec_lab = []

    model.eval()
    with torch.no_grad():
        dict_pred_lab = {}
        i = 0
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device) 
            y_pred, _ = model(inputs)

            if i == 0:
                dict_pred_lab['labels'] = labels
                dict_pred_lab['pred'] = y_pred
                dict_pred_lab['paths'] = paths
                i += 1
            
            #Saving paths and predictions
            dict_pred_lab['labels'] = torch.cat((dict_pred_lab['labels'], labels))
            dict_pred_lab['pred'] = torch.cat((dict_pred_lab['pred'], y_pred))
            dict_pred_lab['paths'] = torch.cat((dict_pred_lab['paths'], paths))

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

        
        #Average precision
        avg_prec_score = comp_average_precision(per_class_avg_prec_pred, per_class_avg_prec_lab, 6)
        mean_avg_prec = sum(avg_prec_score)/len(avg_prec_score) 

    if config['eval'] == True:
        file = open("predictions_testset.pkl", "wb")
        pickle.dump(dict_pred_lab, file)
    
    print(f"----------------- EVALUATION -----------------")
    print(f"Accuracy Per Class:\n{acc_per_class}\nmean accuracy:\n{mean_acc}\n")
    print(f"Average Precision Per Class:\n{avg_prec_score}\nmean AP:\n{mean_avg_prec}\n")
    
    return dict_pred_lab

