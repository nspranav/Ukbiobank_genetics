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

from custom_dataset import CustomDataset
from network import Network

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from utils import *

#%%

parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())

#creating directory

directory = args.job_id
parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs'
path = os.path.join(parent_directory,directory)
model_save_path = os.path.join(path,'models')

if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(model_save_path)

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
batch_size = 50

test_data = CustomDataset(train=False,valid=False)


train_sampler = SubsetRandomSampler(test_data.train_idx)
test_sampler = SubsetRandomSampler(test_data.test_idx)

train_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= train_sampler, num_workers=num_workers)
test_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= test_sampler, num_workers=num_workers)
# %%


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network()

print(model)


#%%

epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=1e-5)

#%%

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

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

    for X,y,_ in train_loader:

        X,y = X.float().to(device),y.to(device)

        actual_train = torch.cat((actual_train,y),0)

        pred = model(X)

        pred_train = torch.cat((pred_train,torch.max(F.softmax(pred,dim=1), dim=1)[1]),0)
        
        loss = criterion(pred,y)

        if torch.isnan(loss):
            break

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #print('loss =',loss.item())

        train_loss += loss.item()
        correct = torch.eq(torch.max(F.softmax(pred,dim=1), dim=1)[1],y).view(-1)
        num_correct_train += torch.sum(correct).item()
    else:
        model.eval()
        valid_loss = 0
        num_correct_valid = 0

        with torch.no_grad():
            for X,y,_ in test_loader:

                X,y = X.float().to(device),y.to(device)

                actual_valid = torch.cat((actual_valid,y),0)

                pred = model(X)
                try:
                    loss = criterion(pred,y)
                except:
                    print(pred)
                    print(y)

                valid_loss += loss.item()
                correct = torch.eq(torch.max(F.softmax(pred,dim=1), dim=1)[1],y).view(-1)
                pred_valid = torch.cat((pred_valid,torch.max(F.softmax(pred,dim=1), dim=1)[1]),0)
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
        

        # plt.figure()
        # plt.plot(actual_valid.detach().cpu().numpy(),pred_valid.detach().cpu()
        #         .numpy(),'.')
        # plt.title('Validation - True vs pred')
        # plt.xlabel('True score')
        # plt.ylabel('Predicted score')
        
        # writer.add_figure('Validation - True vs pred', plt.gcf(),e,True)

        # mcc_t,f1_t,b_a_t = write_confusion_matrix(writer, actual_train.detach().cpu().numpy(),
        #     pred_train.detach().cpu().numpy(), e,'Confusion Matrix - Train' )
        # mcc_v,f1_v,b_a_v = write_confusion_matrix(writer,actual_valid.detach().cpu().numpy(),
        #     pred_valid.detach().cpu().numpy(), e,'Confusion Matrix - Validation')

        print("Epoch: {}/{}.. ".format(e, epochs),
            #   "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
            #   "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
              'Train Accuracy: {:.3f}..'.format(num_correct_train/len(test_data.train_idx)),
              "test Accuracy: {:.3f}..".format(num_correct_valid/len(test_data.test_idx)),
            #   f'mcc_t: {mcc_t:.3f}..',
            #   f'mcc_v: {mcc_v:.3f}..',
            #   f'f1_t: {f1_t:.3f}..',
            #   f'f1_v: {f1_v:.3f}..',
            f'loss: {train_loss:.4f}'
            )
              
        #writer.add_scalar('Train r2', r2_score(pred_train,actual_train),e)
        #writer.add_scalar('Valid r2', r2_score(pred_valid,actual_valid),e)
        #writer.add_scalar('Train Loss', train_loss/len(train_loader),e)
        #writer.add_scalar('Validation Loss', valid_loss/len(valid_loader),e)
        # writer.add_scalar('Train Accuracy',num_correct_train/len(test_data.train_idx),e)
        # writer.add_scalar('Test Accuracy', num_correct_valid/len(test_data.test_idx),e)
        # if abs(valid_loss/len(test_loader) - train_loss/len(train_loader)) < 0.2:
        #    torch.save(model.state_dict(), os.path.join(model_save_path,
        #         'epoch_'+str(e)))
# %%
writer.flush()
writer.close()

# %%

