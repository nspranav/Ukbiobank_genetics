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

from custom_dataset_embed import CustomDataset
from network import Network

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from utils import *
import sys

writer = None
#%%

parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args)
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())

#creating directory

directory = args.job_id
parent_directory = '/data/users3/pnadigapusuresh1/JobOutputs/Genetics'
path = os.path.join(parent_directory,directory)
model_save_path = os.path.join(path,'models')

if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(model_save_path)

writer = SummaryWriter(log_dir=path)

#%%

########################
# Loading the Data #####
########################

########################
# Loading the Data #####
########################

torch.manual_seed(52)
np.random.seed(52)
# number of subprocesses to use for data loading
num_workers = 1
# how many samples per batch to load
batch_size = 1

test_data = CustomDataset(train=False,valid=False)


train_sampler = SubsetRandomSampler(test_data.train_idx)
test_sampler = SubsetRandomSampler(test_data.test_idx)

train_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= train_sampler, num_workers=num_workers)
test_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= test_sampler, num_workers=num_workers)
# %%


device = "cpu"
print("Using {} device".format(device))

model = Network()

print(model)
parent_directory = '/data/users3/pnadigapusuresh1/JobOutputs/Genetics'
load_path = os.path.join(parent_directory,'3805406','models_fold','1','epoch_11')
model.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))

#%%

epochs = 300
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-5)



#%%


model = model.to(device)


# %%
print('Starting to Train...')


for e in range(1,epochs+1):
    model.train()
    train_loss = 0
    num_correct_train = 0

    actual_train = torch.tensor([]).to(device)
    actual_valid = torch.tensor([]).to(device)
    pred_train = torch.tensor([]).to(device)
    pred_valid = torch.tensor([]).to(device)
    pred_actual = torch.tensor([]).to(device)

    for batch,(X,y,_) in enumerate(train_loader):
        print(batch)
        X,y = X.float().to(device),y.to(device)

        actual_train = torch.cat((actual_train,y),0)

        pred = model(X.float())
        soft_max = F.softmax(pred,dim=1)
        pred_train = torch.cat((pred_train,torch.max(soft_max, dim=1)[1]),0)

        try:
            loss = criterion(pred,y)
            #print(loss)
            if torch.isnan(loss):
                print(loss)
                raise Exception()
            

        except:
            print(pred)
            print(X)
            sys.exit(0)

        # layer0_params = torch.cat([x.view(-1) for x in model.all_layers[0].layer.parameters()])
        # layer2_params = torch.cat([x.view(-1) for x in model.all_layers[2].layer.parameters()])
        # l1_regularization = (torch.norm(layer0_params, 1) 
        #                     + torch.norm(layer2_params,1))
                                
        #print('loss =',loss.item())
        #loss += l1_regularization/(len(layer0_params)+len(layer2_params))
        


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
            for X,y,_ in test_loader:

                X,y = X.float().to(device),y.to(device)

                actual_valid = torch.cat((actual_valid,y),0)

                pred = model(X.float())
                try:
                    loss = criterion(pred,y)
                    
                except:
                    print(pred)
                    print(y)

                valid_loss += loss.item()
                soft_max = F.softmax(pred,dim=1)
                correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
                pred_valid = torch.cat((pred_valid,torch.max(soft_max, dim=1)[1]),0)
                num_correct_valid += torch.sum(correct).item()

        # values = {
        #     'actual_train' : actual_train,
        #     'actual_valid' : actual_valid,
        #     'pred_train' : pred_train,
        #     'pred_valid' : pred_valid
        # }

        # if e % 5 == 0:
        #     with open(path + '/arrays'+str(e)+'.pk', 'wb') as f:
        #         pickle.dump(values, f)
        
        #compute the r square

        
        # plt.figure()
        # plt.plot(actual_train.detach().cpu().numpy(),pred_train.detach().cpu()
        #         .numpy(),'.')
        # plt.title('Train - True vs pred')
        # plt.xlabel('True numeric_score')
        # plt.ylabel('Predicted numeric_score')
        
        # writer.add_figure('Train - True vs pred', plt.gcf(),e,True)
        

        #plt.figure()
        # plt.plot(actual_valid.detach().cpu().numpy(),pred_valid.detach().cpu()
        #         .numpy(),'.')
        # plt.title('Validation - True vs pred')
        # plt.xlabel('True score')
        # plt.ylabel('Predicted score')
        
        
        # writer.add_figure('Validation - True vs pred', plt.gcf(),e,True)
        # w1,b1 = model.all_layers[0].layer.parameters()
        # w2,b2 = model.all_layers[2].layer.parameters()

        # if writer:
        #     mcc_t,f1_t,b_a_t,sensitivity_t,specificity_t = write_confusion_matrix(writer, actual_train.detach().cpu().numpy(),
        #         pred_train.detach().cpu().numpy(), e,'Confusion Matrix - Train' )
        #     mcc_v,f1_v,b_a_v, _, _ = write_confusion_matrix(writer,actual_valid.detach().cpu().numpy(),
        #         pred_valid.detach().cpu().numpy(), e,'Confusion Matrix - Validation')

        #     plt.figure()
        #     plt.hist(w1.detach().cpu().view(-1))
        #     writer.add_figure('W1 distribution',plt.gcf(),e,True)
        
        b_a_t = balanced_accuracy_score(actual_train.detach().cpu().numpy(),
                        pred_train.detach().cpu().numpy())
        b_a_v = balanced_accuracy_score(actual_valid.detach().cpu().numpy(),
                        pred_valid.detach().cpu().numpy())
        print("Epoch: {}/{}.. ".format(e, epochs),
            #   "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
            #   "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
              'Train Accuracy: {:.3f}..'.format(num_correct_train/len(test_data.train_idx)),
              "test Accuracy: {:.3f}..".format(num_correct_valid/len(test_data.test_idx)),
              f'train_bal: {b_a_t:.3f}',
              f'test_bal: {b_a_v:.3f}',
            #   f'mcc_t: {mcc_t:.3f}..',
            #   f'mcc_v: {mcc_v:.3f}..',
            #   f'f1_t: {f1_t:.3f}..',
            #   f'f1_v: {f1_v:.3f}..',
            f'loss: {train_loss:.4f}'
            )

        if b_a_t - b_a_v > 0.1:
            print(f'Bye')
            exit() 
              
        #writer.add_scalar('Train r2', r2_score(pred_train,actual_train),e)
        #writer.add_scalar('Valid r2', r2_score(pred_valid,actual_valid),e)
        #writer.add_scalar('Train Loss', train_loss/len(train_loader),e)
        #writer.add_scalar('Validation Loss', valid_loss/len(valid_loader),e)
        # writer.add_scalar('Train Accuracy',num_correct_train/len(test_data.train_idx),e)
        # writer.add_scalar('Test Accuracy', num_correct_valid/len(test_data.test_idx),e)
        if abs(valid_loss/len(test_loader) - train_loss/len(train_loader)) < 0.2:
           torch.save(model.state_dict(), os.path.join(model_save_path,
                'epoch_'+str(e)))
# %%
writer.flush()
writer.close()

# %%

