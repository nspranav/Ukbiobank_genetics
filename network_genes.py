# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import math
from torch.nn import init
from torch.nn.parameter import Parameter

from contrastive_loss import ContrastiveLoss
from supervised_contrastive_loss import SupervisedContrastiveLoss

import lightning as L

# %%

class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Linear(in_features=1060,out_features=400)
        self.second_layer = nn.Linear(in_features=400, out_features=600)
        self.third_layer = nn.Linear(in_features=600,out_features=400)
        self.fourth_layer = nn.Linear(in_features=400,out_features=300)
        self.fifth_layer = nn.Linear(in_features=300,out_features=150)
        self.layer6 = nn.Linear(150,100)
        self.layer7 = nn.Linear(100,32)
        dropout = nn.Dropout(0.2)
        self.all_layers = nn.Sequential(
                                        self.first_layer,nn.ReLU6(),dropout,
                                        self.second_layer,nn.ReLU6(),dropout,
                                        self.third_layer,nn.ReLU(), dropout,
                                        self.fourth_layer,nn.ReLU6(), dropout,
                                        self.fifth_layer,nn.ReLU6(), dropout,
                                        self.layer6)
        

    def forward(self, gene,data=None):
        # self.first_layer.weight
        img = self.all_layers(gene)

        return img



# %%
class ContrastiveLearning(L.LightningModule):
    def __init__(self, smri_network=None,gene_network=None,
                 lambda_smri_gene = 0.3,
                 lambda_gene_smri = 0.7,
                 lambda_smri_smri=0,
                 lambda_gene_gene=0, 
                 tau = 0.5,learning_rate = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['smri_network','gene_network'])
        #self.automatic_optimization=False

        self.smri_NN = smri_network
        self.gene_NN = gene_network

        self.loss_fn = ContrastiveLoss(temperature=tau)
        self.sup_loss_fn = SupervisedContrastiveLoss(temperature=tau)

        self.lambda_smri_gene = lambda_smri_gene
        self.lambda_gene_smri = lambda_gene_smri
        self.lambda_smri_smri = lambda_smri_smri
        self.lambda_gene_gene = lambda_gene_gene
        self.tau = tau
        self.lr = learning_rate


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X_genes,X_smri,y,_ = batch
        pred_genes = self.gene_NN(X_genes.float())
        pred_smri = self.smri_NN(X_smri.float())

        loss_smri_gene,loss_gene_smri = self.loss_fn(pred_smri,pred_genes)
        loss_smri_smri = self.sup_loss_fn(pred_smri,y)
        loss_gene_gene = self.sup_loss_fn(pred_genes,y)
        loss = (self.lambda_smri_gene *loss_smri_gene  
                + (self.lambda_gene_smri) * loss_gene_smri
                + (self.lambda_smri_smri)* loss_smri_smri
                + (self.lambda_gene_gene)* loss_gene_gene
                )

        values = {'train_loss_smri_gene': self.lambda_smri_gene *loss_smri_gene,
                  'train_loss_gene_smri': (self.lambda_gene_smri) * loss_gene_smri,
                  'train_loss_smri_smri': (self.lambda_smri_smri)* loss_smri_smri,
                  'train_loss_gene_gene': (self.lambda_gene_gene)*loss_gene_gene,
                  'train_loss': loss}
        #print(values)
        self.log_dict(values)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X_genes,X_smri,y,_ = batch
        pred_genes = self.gene_NN(X_genes.float())
        pred_smri = self.smri_NN(X_smri.float())

        loss_smri_gene,loss_gene_smri = self.loss_fn(pred_smri,pred_genes)
        loss_smri_smri = self.sup_loss_fn(pred_smri,y)
        loss_gene_gene = self.sup_loss_fn(pred_genes,y)
        loss = ((self.lambda_smri_gene *loss_smri_gene)  
                + (self.lambda_gene_smri) * loss_gene_smri
                + (self.lambda_smri_smri)* loss_smri_smri
                + (self.lambda_gene_gene)*loss_gene_gene
                )

        values = {'valid_loss_smri_gene': self.lambda_smri_gene *loss_smri_gene,
                  'valid_loss_gene_smri': (self.lambda_gene_smri) * loss_gene_smri,
                  'valid_loss_smri_smri': (self.lambda_smri_smri)* loss_smri_smri,
                  'valid_loss_gene_gene': (self.lambda_gene_gene)*loss_gene_gene,
                  'val_loss': loss}
        self.log_dict(values)

    def test_step(self, batch, batch_idx):
        X_genes,X_smri,y,_ = batch
        pred_genes = self.gene_NN(X_genes.float())
        pred_smri = self.smri_NN(X_smri.float())

        loss_smri_gene,loss_gene_smri = self.loss_fn(pred_smri,pred_genes)
        loss_smri_smri = self.sup_loss_fn(pred_smri,y)
        loss_gene_gene = self.sup_loss_fn(pred_genes,y)
        loss = ((self.lambda_smri_gene *loss_smri_gene)  
                + (self.lambda_gene_smri) * loss_gene_smri
                + (self.lambda_smri_smri)* loss_smri_smri
                + (self.lambda_gene_gene)*loss_gene_gene
                )

        values = {'test_loss_smri_gene': self.lambda_smri_gene *loss_smri_gene,
                  'test_loss_gene_smri': (self.lambda_gene_smri) * loss_gene_smri,
                  'test_loss_smri_smri': (self.lambda_smri_smri)* loss_smri_smri,
                  'test_loss_gene_gene': (self.lambda_gene_gene)*loss_gene_gene,
                  'test_loss': loss}
        self.log_dict(values)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=self.lr,
                                    weight_decay=1e-5
                                    )
        return optimizer
