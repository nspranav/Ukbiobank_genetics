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
                    , label_file= '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/subset_vars.csv',transform=None
                    , target_transform=None,train=True,valid=False,random_state=52):
        self.img_path = img_path
        # self.dirs = os.listdir(img_path)
        # # The below list contains the index of directories that are missing the
        # # mri images. 
        # self.no_smri = [8073,12877,15213,18350,29723] #indices
        # #[1336082,2464107, 4158450, 3429974, 4468630] subject ids
        # self.dirs.pop(8073)
        # self.dirs.pop(12876)
        # self.dirs.pop(15211)
        # self.dirs.pop(18347)
        # self.dirs.pop(29719)
        # self.dirs.remove('Raw_T1.m')
        """
        # column for age: age_when_attended_assessment_centre_f21003_0_0
        # column for sex; sex_f31_0_0 
        # column for Numeric memory: 'maximum_digits_remembered_correctly_f4282_2_0'
        """

        self.train = train
        self.valid = valid

        self.vars = pd.read_csv(label_file,index_col='eid',
                            usecols=['eid','maximum_digits_remembered_correctly_f4282_2_0',
                            'sex_f31_0_0','age_when_attended_assessment_centre_f21003_0_0'])
        self.vars.columns = ['sex','score','age']

        self.gene_dataset = pd.read_csv('gene_working_mem_dataset.csv',index_col='FID')
        self.gene_dataset = self.gene_dataset.fillna(0)

        #self.pca_smri = pd.read_csv('PCA_transformed_smri.csv',index_col='dirs')
        self.img_vars = self.vars
        #ids with no images
        self.vars = self.vars.drop([1171080,1660210,2012720,2378544,2835040,2951207,4312676],axis=0)
        self.gene_dataset = self.gene_dataset.drop([1171080,1660210,2012720,2378544,2835040,2951207,4312676],axis=0)
        
        #selecting only subjects with genes
        #self.pca_smri = self.pcs_smri.loc[self.gene_dataset.index.tolist()]
        #self.vars['score'] = self.vars['score'] + 1 
        
        # Applying log transform
        #self.vars = self.vars.apply(np.log,axis=1)

        #self.vars['score'] = SimpleImputer(strategy='mean',
        #                       missing_values=np.nan).fit_transform(self.vars)
        
        self.vars = self.vars.loc[self.gene_dataset.index.tolist()]
        # removing missing scores 
        self.vars = self.vars.loc[
                        self.vars.score.isin([2,3,4,5,9,10,11,12])]
        self.img_vars = self.img_vars.loc[
                        self.img_vars.score.isin([2,3,4,5,9,10,11,12])]

        #######
        self.vars['new_score'] = [0 if a < 6 else 1 for a in self.vars['score']]
        self.img_vars['new_score'] = [0 if a < 6 else 1 for a in self.img_vars['score']]
        # We want sampling only during the training and validation and not for
        # testing the fixed model
        
        # if not test:
        #     maj_class = resample(self.vars[self.vars.new_score == 0],
        #             n_samples = 2250,replace=False,random_state=random_state)
        #     min_class = self.vars[self.vars.new_score == 1]
        #     self.vars = pd.concat([maj_class,min_class])

        # Need to sort by index because we want the order of data same for
        # both the train and validation dataset
        #self.vars.drop([1336082,2464107, 4158450, 3429974, 4468630],inplace=True)
                
        #######

        # Removing the images with no pixels
        #self.vars = self.vars.drop([1171080,1660210,2012720,2378544,2835040,2951207,4312676],axis=0)
        

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        train_idx, test_idx = next(sss.split(np.zeros_like(self.img_vars),
            self.img_vars.new_score.values))
        img_train = self.img_vars.iloc[train_idx]
        img_test = self.img_vars.iloc[test_idx]
        train_vars = self.vars.loc[list(set(img_train.index.tolist()).intersection(self.vars.index.tolist()))]
        self.train_idx = list(range(4494))
        self.test_idx = list(range(4494,4995))
        test_vars = self.vars.loc[list(set(img_test.index.tolist()).intersection(self.vars.index.tolist()))]
        self.vars = pd.concat([train_vars,test_vars])
        self.gene_dataset = self.gene_dataset.loc[self.vars.index.tolist()]
        self.dirs = self.vars.index.tolist()

        if train or valid:
            self.vars = self.vars.iloc[self.train_idx]
            self.gene_dataset = self.gene_dataset.iloc[self.train_idx]
        else:
            self.test_vars = self.vars.iloc[self.test_idx]
            self.test_gene_dataset = self.gene_dataset.iloc[self.test_idx]
            # self.female_idx = self.test_vars[self.test_vars.sex == 0].pos.tolist()
            # self.male_idx = self.test_vars[self.test_vars.sex==1].pos.tolist()



        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self,idx):
        
        label = self.vars.iloc[idx]
        gene = self.gene_dataset.iloc[idx].to_numpy()



        #offset by 4 because of scores range from 4 to 9
        return torch.tensor(gene),int(label['new_score']),label['age']
# %%



