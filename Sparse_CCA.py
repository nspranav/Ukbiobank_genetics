############

#Check the google colab sparse_cca.ipynb


#############
#%%
import numpy as np
import argparse
import subprocess

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import glob

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from custom_dataset_embed import CustomDataset
from network_genes import Network as Network_g,ContrastiveLearning
from network_smri import Network as Network_smmri


from sklearn.model_selection import StratifiedShuffleSplit

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--job_id',type=str,required=False)
# 5379443
parser.add_argument('--load_from',type=str,required=False,default='5416836')
args,_ = parser.parse_known_args()
print(args)
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())

# %%

torch.manual_seed(52)
np.random.seed(52)
# number of subprocesses to use for data loading
num_workers = 5
# how many samples per batch to load
batch_size = 5000

test_data = CustomDataset(train=False,valid=False)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=52)
train_idx, valid_idx = next(sss.split(np.zeros_like(test_data.vars.iloc[test_data.train_idx]),
            test_data.vars.iloc[test_data.train_idx].new_score.values))


train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(np.hstack(valid_idx))
test_sampler = SubsetRandomSampler(test_data.test_idx)

train_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= train_sampler, num_workers=num_workers)
valid_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= valid_sampler, num_workers=num_workers)
test_loader = DataLoader(test_data,batch_size=batch_size, 
                            sampler= test_sampler, num_workers=num_workers)

#%%
smri_model = Network_smmri()
gene_model = Network_g()
# model = ContrastiveLearning(smri_model,gene_model,
#                             tau=1,
#                             lambda_smri_gene=0.7,
#                             lambda_gene_smri= 0.3,
#                             lambda_smri_smri= 0.1,
#                             lambda_gene_gene= 0.9,
#                             learning_rate= 1e-4)



check_point = glob.glob(f'./tb_logs/{args.load_from}/version_0/checkpoints/*.ckpt')[0]

model = ContrastiveLearning.load_from_checkpoint(check_point,
                                                smri_network = smri_model,
                                                gene_network = gene_model
                                            )
# %%
train_repr_smri = torch.tensor([],device=model.device)
train_repr_gene = torch.tensor([],device=model.device)
valid_repr_smri = torch.tensor([],device=model.device)
valid_repr_gene = torch.tensor([],device=model.device)
test_repr_smri  = torch.tensor([],device=model.device)
test_repr_gene  = torch.tensor([],device=model.device)

for X_gene,X_smri, y, _ in train_loader:
    pred_smri = model.smri_NN(X_smri.to(model.device).float())
    pred_gene = model.gene_NN(X_gene.to(model.device).float())
    train_repr_smri = torch.cat([train_repr_smri,pred_smri])
    train_repr_gene = torch.cat([train_repr_gene,pred_gene])

for X_gene,X_smri, y, _ in valid_loader:
    pred_smri = model.smri_NN(X_smri.to(model.device).float())
    pred_gene = model.gene_NN(X_gene.to(model.device).float())
    valid_repr_gene = torch.cat([valid_repr_gene,pred_gene])
    valid_repr_smri = torch.cat([valid_repr_smri,pred_smri])

for X_gene,X_smri, y, _ in test_loader:
    pred_smri = model.smri_NN(X_smri.to(model.device).float())
    pred_gene = model.gene_NN(X_gene.to(model.device).float())
    test_repr_gene = torch.cat([test_repr_gene,pred_gene])
    test_repr_smri = torch.cat([test_repr_smri,pred_smri])

# %%
# with open('5216125_L1L2_repr.npy', 'wb') as f:
#     np.save(f, train_repr_smri.detach().cpu().numpy() )
#     np.save(f, train_repr_gene.detach().cpu().numpy() )
#     np.save(f, valid_repr_smri.detach().cpu().numpy() )
#     np.save(f, valid_repr_gene.detach().cpu().numpy() )
#     np.save(f,  test_repr_smri.detach().cpu().numpy() )
#     np.save(f,  test_repr_gene.detach().cpu().numpy() )
# %%
# with open('5216125_L1L2_repr.npy', 'wb') as f:
#     train_repr_smri = np.load(f)
#     train_repr_gene = np.load(f)
#     valid_repr_smri = np.load(f)
#     valid_repr_gene = np.load(f)
#     test_repr_smri = np.load(f)
#     test_repr_gene = np.load(f)

#%%
X_train = train_repr_smri.detach().cpu().numpy()
Z_train = train_repr_gene.detach().cpu().numpy()
X_valid = valid_repr_smri.detach().cpu().numpy()
Z_valid = valid_repr_gene.detach().cpu().numpy()
X_test = test_repr_smri.detach().cpu().numpy()
Z_test = test_repr_gene.detach().cpu().numpy()
# %%

import numpy as np
import pandas as pd
from scipy import stats

#%%

df1 = pd.DataFrame(X_train)
df2 = pd.DataFrame(Z_train)
indices = (np.abs(stats.zscore(df1)) < 2.5).all(axis=1)
df1 = df1[indices]
df2 = df2[indices]
indices = (np.abs(stats.zscore(df2)) < 2.5).all(axis=1)
X_train = df1[indices]
Z_train = df2[indices]
print(df1.shape,df2.shape)

#%%
df1 = pd.DataFrame(X_valid)
df2 = pd.DataFrame(Z_valid)
indices = (np.abs(stats.zscore(df1)) < 2.5).all(axis=1)
df1 = df1[indices]
df2 = df2[indices]
indices = (np.abs(stats.zscore(df2)) < 2.5).all(axis=1)
X_valid = df1[indices]
Z_valid = df2[indices]
print(df1.shape,df2.shape)

#%%
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Scale the data
X_train = scaler_X.fit_transform(X_train)
Z_train = scaler_Y.fit_transform(Z_train)
X_valid = scaler_X.fit_transform(X_valid)
Z_valid = scaler_Y.fit_transform(Z_valid)
X_test = scaler_X.fit_transform(X_test)
Z_test = scaler_Y.fit_transform(Z_test)

from sparse_cca import cca_pmd

# %%
def get_reslts(pmd,dim):
    x_weights = pmd.best_estimator_.weights_[0][:,dim]
    z_weights = pmd.best_estimator_.weights_[1][:,dim]
    train = np.corrcoef(np.dot(x_weights, X_train.T), np.dot(z_weights, Z_train.T))[0, 1]
    valid = np.corrcoef(np.dot(x_weights, X_valid.T), np.dot(z_weights, Z_valid.T))[0, 1]
    test = np.corrcoef(np.dot(x_weights, X_test.T), np.dot(z_weights, Z_test.T))[0, 1]
    return train,valid,test

#%%
from cca_zoo.linear import SPLS,SCCA_IPLS,ElasticCCA,SCCA_Span
from cca_zoo.model_selection import GridSearchCV
from joblib import dump

param_grid_pmd = {"tau": [[0.1,0.2,0.8,1],[0.1,0.2,0.8,1]]}
pmd = GridSearchCV(
    SPLS(epochs=15, early_stopping=True,latent_dimensions=5), param_grid=param_grid_pmd
).fit([X_train, Z_train])

# %%
print(pmd.best_estimator_.get_params())
print(get_reslts(pmd,0))
print(get_reslts(pmd,1))
print(get_reslts(pmd,2))
print()
# %%
