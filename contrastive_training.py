#%%
import numpy as np
import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler


from custom_dataset_embed import CustomDataset
from network_genes import Network as Network_g
from network_smri import Network as Network_smmri

from torch.utils.tensorboard import SummaryWriter
from utils import *
import sys

from contrastive_loss import ContrastiveLoss

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
batch_size = 50

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

model_genes = Network_g()
model_smri = Network_smmri()

print(model_genes)
print(model_smri)
parent_directory = '/data/users3/pnadigapusuresh1/JobOutputs/Genetics'
load_path = os.path.join(parent_directory,'3805406','models_fold','1','epoch_11')
#model.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))

#%%

epochs = 300
criterion = nn.CrossEntropyLoss()
optimizer_genes = optim.Adam(params=model_genes.parameters(), lr=1e-5)
optimizer_smri = optim.Adam(params=model_smri.parameters(), lr=1e-5)



#%%


model_genes = model_genes.to(device)
model_smri = model_smri.to(device)

loss_fn = ContrastiveLoss(temperature=0.5)

# %%
print('Starting to Train...')

model_genes.train()
model_smri.train()
for e in range(1,epochs+1):
    train_loss = 0
    valid_loss = 0
    num_correct_train = 0
    
    tau = 1
    lambda_smri_gene = 0.5 
    for batch,(X_genes,X_smri,y,_) in enumerate(train_loader):
        X_genes,X_smri,y = X_genes.float().to(device),X_smri.float().to(device),y.to(device)

        pred_genes = model_genes(X_genes.float())
        pred_smri = model_smri(X_smri.float())
      
        loss_smri_gene,loss_gene_smri = loss_fn(pred_smri,pred_genes)
        loss = (lambda_smri_gene *loss_smri_gene) + (1 - lambda_smri_gene) * loss_gene_smri
        
        if torch.isnan(loss):
            print(loss)
            sys.exit(-1)
        
        train_loss += loss.item()

        # Backpropagation
        optimizer_smri.zero_grad()
        optimizer_genes.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer_smri.step()
        optimizer_genes.step()
        
        #print('l1 reg = ',l1_regularization)
    for batch,(X_genes,X_smri,y,_) in enumerate(test_loader):
        X_genes,X_smri,y = X_genes.float().to(device),X_smri.float().to(device),y.to(device)

        pred_genes = model_genes(X_genes.float())
        pred_smri = model_smri(X_smri.float())
      

        try:

            loss_smri_gene,loss_gene_smri = loss_fn(pred_smri,pred_genes)
            loss = (lambda_smri_gene *loss_smri_gene) + (1 - lambda_smri_gene) * loss_gene_smri

            if torch.isnan(loss):
                print(loss)
                sys.exit(-1)
            
            valid_loss += loss.item()
        except:
            print(pred_genes)
            print(pred_smri)
            print(X_genes)
            sys.exit(0)
        
    print(f'(Epoch {e}:) train loss  = {train_loss/len(test_data.train_idx)}, valid loss  = {valid_loss/len(test_data.test_idx)}')




# %%
writer.flush()
writer.close()

# %%

