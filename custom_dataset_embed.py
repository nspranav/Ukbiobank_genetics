#%%
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms
import random
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
#%%

class CustomDataset(Dataset):
    
    def __init__(self,img_path= '/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data'
                    , label_file= '/data/users3/pnadigapusuresh1/Projects/ukbiobank/Data/subset_vars.csv',transform=None
                    , target_transform=None,train=True,valid=False,random_state=52):
        """
        # column for age: age_when_attended_assessment_centre_f21003_0_0
        # column for sex; sex_f31_0_0 
        # column for Numeric memory: 'maximum_digits_remembered_correctly_f4282_2_0'
        """
        self.img_path = img_path
        self.train = train
        self.valid = valid

        self.vars = pd.read_csv(label_file,index_col='eid',
                            usecols=['eid','maximum_digits_remembered_correctly_f4282_2_0',
                            'sex_f31_0_0','age_when_attended_assessment_centre_f21003_0_0'])
        self.vars.columns = ['sex','score','age']

        #self.gene_dataset = pd.read_csv('smri_gene_embed.csv',index_col=0)
        self.gene_dataset = pd.read_csv('smri_embed_raw_SNP.csv',index_col=0)
        self.gene_dataset = self.gene_dataset.fillna(1)
        
        self.vars = self.vars.loc[self.gene_dataset.index.tolist()]
        self.vars['new_score'] = [0 if a < 6 else 1 for a in self.vars['score']]

        self.vars = self.vars.sort_index()
        self.gene_dataset = self.gene_dataset.sort_index()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=random_state)
        self.train_idx, self.test_idx = next(sss.split(np.zeros_like(self.vars),
            self.vars.new_score.values))

        self.train_idx = np.hstack([self.train_idx])
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    
    def __len__(self):
        return len(self.vars)

    def __getitem__(self,idx):
        
        label = self.vars.iloc[idx]
        smri = self.gene_dataset.iloc[idx,:512].to_numpy() #smri
        gene = self.gene_dataset.iloc[idx,512:].to_numpy() #genes




        #offset by 4 because of scores range from 4 to 9
        return torch.tensor(gene),torch.tensor(smri),int(label['new_score']),label['age']
# %%



