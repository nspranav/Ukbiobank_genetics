#%%
import numpy as np
import pandas as pd
import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from custom_dataset import CustomDataset
from network import Network
from utils import *

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


#%%
torch.set_default_dtype(torch.float32)
parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
parser.add_argument('array_id',type=int)
args = parser.parse_args()
print(args.job_id, args.array_id)
print('number of gpus ',torch.cuda.device_count())

#creating directory

directory = args.job_id
parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs/Genetics'
path = os.path.join(parent_directory,directory)
model_save_path = os.path.join(path,'models_fold')

if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(model_save_path)

#%%
lambda0 = np.flip(np.linspace(1e-3,1e-7,30))
lambda1 = np.flip(np.linspace(1e-3,1e-7,30))
lr = np.linspace(1e-2,1e-7,30)

combinations = [[x0, y0, l] for x0 in lambda0 for y0 in lambda1 for l in lr]

lr = lr[args.array_id - 1]

print(f'lr = {lr}')

count_09 = 0
#%%

########################
# Loading the Data #####
########################



torch.manual_seed(52)
# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 1000
# if torch.cuda.device_count() > 1:
#     batch_size *= torch.cuda.device_count()
# else:
#     batch_size = 5
# percentage of training set to use as validation
valid_size = 0.30
# percentage of data to be used for testset
test_size = 0.10


data = CustomDataset(train=False,valid=False)

# get filtered variables
vars = data.vars.iloc[data.train_idx]

#%% 

# Prepare for k-fold

sss = StratifiedShuffleSplit(n_splits=1,test_size=valid_size,random_state=52)
fold = 1

for train_idx, valid_idx in sss.split(np.zeros_like(vars),vars.new_score.values):
    writer = SummaryWriter(log_dir=path+'/fold'+str(fold))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(data.test_idx)

    train_loader = DataLoader(data,batch_size=batch_size, 
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(data,batch_size=batch_size, 
                                sampler= valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(data,batch_size=batch_size,
                                sampler= test_sampler, num_workers=num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {} device".format(device))

    model = Network()
    print(model)


    #%%

    epochs = 500
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.1)
    #%%

    # Regularization terms

    model.to(device)

    print('Starting to Train...')

    for e in range(1,epochs+1):
        model.train()
        train_loss = 0
        num_correct_train = 0

        actual_train = torch.tensor([]).to(device)
        actual_valid = torch.tensor([]).to(device)
        actual_test = torch.tensor([]).to(device)
        pred_train = torch.tensor([]).to(device)
        pred_valid = torch.tensor([]).to(device)
        pred_test = torch.tensor([]).to(device)


        for X,y,_ in train_loader:

            X,y = X.float().to(device),y.to(device)
            #print(actual_train.dtype,y.dtype,actual_train.shape,y.shape)
            actual_train = torch.cat((actual_train,y),0)

            pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
            soft_max = F.softmax(pred,dim=1)
            pred_train = torch.cat((pred_train,torch.max(soft_max, dim=1)[1]),0)

            try:
                loss = criterion(pred,y)
                #print(loss)
                if torch.isnan(loss):
                            print(loss)
                            raise Exception()
                

            except:
                print(loss)
                print(pred)
                print(X)
                exit(0)

            layer0_params = torch.cat([x.view(-1) for x in model.all_layers[0].layer.parameters()])
            layer2_params = torch.cat([x.view(-1) for x in model.all_layers[2].layer.parameters()])
            l1_regularization = (lambda0 * torch.norm(layer0_params, 1).item()
                                + lambda1* torch.norm(layer2_params,1).item())
                                    
            #print('loss =',loss.item())
            loss += l1_regularization + l1_regularization
            


            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            
            
            #print('l1 reg = ',l1_regularization)

            train_loss += loss.item()
            correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
            num_correct_train += torch.sum(correct).item()
        else:
            model.eval()
            valid_loss = 0
            num_correct_valid = 0

            with torch.no_grad():
                for X,y,_ in valid_loader:

                    X,y = X.float().to(device),y.to(device)

                    actual_valid = torch.cat((actual_valid,y),0)

                    pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
                    try:
                        loss = criterion(pred,y)
                        
                    except:
                        print('valid',loss)
                        print(pred)
                        print(y)

                    valid_loss += loss.item()
                    soft_max = F.softmax(pred,dim=1)
                    correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
                    pred_valid = torch.cat((pred_valid,torch.max(soft_max, dim=1)[1]),0)
                    num_correct_valid += torch.sum(correct).item()
        model.eval()
        test_loss = 0
        num_correct_test = 0

        with torch.no_grad():
            for X,y,_ in test_loader:

                X,y = X.float().to(device),y.to(device)

                actual_test = torch.cat((actual_test,y),0)

                pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
                try:
                    loss = criterion(pred,y)
                    
                except:
                    print(pred)
                    print(y)

                test_loss += loss.item()
                soft_max = F.softmax(pred,dim=1)
                correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
                pred_test = torch.cat((pred_test,torch.max(soft_max, dim=1)[1]),0)
                num_correct_test += torch.sum(correct).item()

        mcc_t,f1_t,b_a_t,specificity_t,sensitivity_t = write_confusion_matrix(writer, actual_train.detach().cpu().numpy(),
            pred_train.detach().cpu().numpy(), e,'Confusion Matrix - Train' )
        mcc_v,f1_v,b_a_v,specificity_v,sensitivity_v = write_confusion_matrix(writer,actual_valid.detach().cpu().numpy(),
            pred_valid.detach().cpu().numpy(), e,'Confusion Matrix - Validation')
        mcc_test,f1_test,b_a_test,specificity_test,sensitivity_test = write_confusion_matrix(writer,actual_test.detach().cpu().numpy(),
            pred_test.detach().cpu().numpy(), e,'Confusion Matrix - Test')
        
        # auc_train = roc_auc_score(actual_train.detach().cpu().numpy(),pred_proba_train.detach().cpu().numpy())
        # auc_valid = roc_auc_score(actual_valid.detach().cpu().numpy(),pred_proba_valid.detach().cpu().numpy())
        # auc_test = roc_auc_score(actual_test.detach().cpu().numpy(),pred_proba_test.detach().cpu().numpy())  
        
        print("Epoch: {}/{}.. ".format(e, epochs),
            "Training Accuracy: {:.3f}.. ".format(num_correct_train/len(train_idx)),
            "Validation Accuracy: {:.3f}.. ".format(num_correct_valid/len(valid_idx)),
            "Test Accuracy: {:.3f}.. ".format(num_correct_test/len(data.test_idx)),
            "Train Bal acc: {:.4f}.. ".format(b_a_t),
            "Val Bal acc: {:.4f}.. ".format(b_a_v),
            "Test Bal acc: {:.4f}.. ".format(b_a_test),
            f'loss: {train_loss:.4f}'
        )
            
        # writer.add_scalar('Train r2', r2_score(pred_train,actual_train),e)
        # writer.add_scalar('Valid r2', r2_score(pred_valid,actual_valid),e)
        writer.add_scalar('Train Loss', train_loss/len(train_loader),e)
        writer.add_scalar('Validation Loss', valid_loss/len(valid_loader),e)
        writer.add_scalar('Train Accuracy',num_correct_train/len(train_idx),e)
        writer.add_scalar('validation Accuracy', num_correct_valid/len(valid_idx),e)
        
        # w1,b1 = model.all_layers[0].layer.parameters()
        # w2,b2 = model.all_layers[2].layer.parameters()

        # plt.figure()
        # plt.hist(w1.detach().cpu().view(-1))
        # writer.add_figure('W1 distribution',plt.gcf(),e,True)
        
        # if abs(valid_loss/len(valid_loader) - train_loss/len(train_loader)) < 0.2:
        #     fold_path = os.path.join(model_save_path,str(fold))

        #     if not os.path.exists(fold_path):
        #         os.makedirs(fold_path)
        #     torch.save(model.state_dict(), os.path.join(fold_path,
        #         'epoch_'+str(e)))

        if b_a_t > 0.9:
            count_09 += 1

            if count_09 == 20:
                print('Bye')
                exit(0)
        
        scheduler.step()

    fold+=1
    print('####################################################################')
    writer.flush()
    writer.close()

# %%

