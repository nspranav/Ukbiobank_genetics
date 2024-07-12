#%%
import numpy as np
import argparse
import glob

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from custom_dataset_embed import CustomDataset
from stage3_network import Network as net
from stage3_network import ContrastiveClassification

from sklearn.model_selection import StratifiedShuffleSplit

import wandb
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--job_id',type=str,required=False)
parser.add_argument('--load_from',type=str,required=False)
parser.add_argument('--group',type=str,required=False)

args,_ = parser.parse_known_args()
print(args)
print(args.job_id)
print('loading from',args.load_from)
print('number of gpus ',torch.cuda.device_count())


# %%

L.seed_everything(52)
# number of subprocesses to use for data loading
num_workers = 50
# how many samples per batch to load


test_data = CustomDataset(train=False,valid=False)

#%%
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=52)
for i,(train_idx, valid_idx) in enumerate(sss.split(np.zeros_like(test_data.vars.iloc[test_data.train_idx]),
            test_data.vars.iloc[test_data.train_idx].new_score.values)):
    
    # Job 5156841 - 1st loss function
    # Job 5201695 - 3rd and 4th
    job_to_load_from = args.load_from if args.load_from is not None else '5216125'

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(np.hstack([valid_idx]))
    test_sampler = SubsetRandomSampler(test_data.test_idx)
    
    batch_size = 25
    train_loader = DataLoader(test_data,batch_size=batch_size, 
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(test_data,batch_size=batch_size, 
                                sampler= valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data,batch_size=batch_size, 
                                sampler= test_sampler, num_workers=num_workers)


    logger = WandbLogger(save_dir="./WB_classification_logs/",name=job_to_load_from+f'_{i+1}',
                         group=args.group, project='Classification')

    model = ContrastiveClassification(network=net(),load_from= job_to_load_from,
                                    learning_rate=1e-5)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, 
                                        monitor="valid_acc",
                                        mode='max',
                                        filename='epoch{epoch:02d}-valid_acc{valid_acc:.2f}')

    early_stopping_callback = EarlyStopping(monitor='valid_acc',
                                            mode='max',
                                            patience=35,
                                            verbose=True)
    trainer = L.Trainer(max_epochs=1000,
                        callbacks=[checkpoint_callback,
                                early_stopping_callback],
                        accumulate_grad_batches=1,
                        log_every_n_steps=5,
                        logger=logger)
    
    trainer.fit(model,train_loader,valid_loader)
    
    experiment_id = logger.experiment.id
    
    #trainer.test(dataloaders=test_loader,ckpt_path="best")
    wandb.finish()
    logger2 = WandbLogger(save_dir="./WB_results/",name=job_to_load_from+f'_{i+1}',
                         group=args.group, project='classification_results')
    check_point = glob.glob(f'./WB_classification_logs/Classification/{experiment_id}/checkpoints/*.ckpt')[0]
    model = ContrastiveClassification.load_from_checkpoint(check_point,network=net(),load_from= job_to_load_from,
                                    learning_rate=1e-4)
    trainer = L.Trainer(logger=logger2)

    batch_size = 5000
    
    train_loader = DataLoader(test_data,batch_size=batch_size, 
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(test_data,batch_size=batch_size, 
                                sampler= valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data,batch_size=batch_size, 
                                sampler= test_sampler, num_workers=num_workers)

    trainer.test(model=model,dataloaders=[train_loader,valid_loader,test_loader])
    wandb.finish()
#%%
import subprocess
subprocess.run(['python', 'calculate_metrics.py', '--group', args.group, '--job_id',args.load_from])