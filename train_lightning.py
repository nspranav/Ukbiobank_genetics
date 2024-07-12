#%%
import numpy as np
import argparse
import subprocess

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


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
parser.add_argument('--group',type=str,required=False)
args,_ = parser.parse_known_args()
print(args)
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())

#%%
#creating directory

directory = args.job_id
print(f'{directory}')
logger = TensorBoardLogger(save_dir="./tb_logs/", name=directory if directory is not None else 'test')


# %%

torch.manual_seed(52)
np.random.seed(52)
# number of subprocesses to use for data loading
num_workers = 5
# how many samples per batch to load
batch_size = 50

test_data = CustomDataset(train=False,valid=False)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=52)
train_idx, valid_idx = next(sss.split(np.zeros_like(test_data.vars.iloc[test_data.train_idx]),
            test_data.vars.iloc[test_data.train_idx].new_score.values))


train_sampler = SubsetRandomSampler(np.hstack([train_idx,valid_idx]))
valid_sampler = SubsetRandomSampler(valid_idx)
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
model = ContrastiveLearning(smri_model,gene_model,
                            tau=0.5,
                            lambda_smri_gene= 0.8,
                            lambda_gene_smri= 0.2,
                            lambda_smri_smri= 0,
                            lambda_gene_gene= 1,
                            learning_rate= 1e-4)

# model = ContrastiveLearning.load_from_checkpoint('./tb_logs/5216125/version_0/checkpoints/epoch=17-step=2430.ckpt',
#                                                 smri_network = smri_model,
#                                                 gene_network = gene_model,
#                                                 tau=0.7,
#                                                 lambda_smri_gene=0,
#                                                 lambda_gene_smri= 0,
#                                                 lambda_smri_smri= 1,
#                                                 lambda_gene_gene= 0,
#                                                 learning_rate= 1e-5)
# %%
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        #min_delta=1e-3,
                                        patience=10,
                                        verbose=True)
trainer = L.Trainer(max_epochs=150,
                    callbacks=[checkpoint_callback,
                               early_stopping_callback],
                    accumulate_grad_batches=1,
                    log_every_n_steps=5,
                    logger=logger)
# %%
trainer.fit(model,train_loader,valid_loader)

# %%
subprocess.run(['python', 'lightning_classification.py', '--load_from', 
                args.job_id, '--group', args.group])