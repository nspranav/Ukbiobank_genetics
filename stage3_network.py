# %%

import torch
import torch.nn as nn
import torch.nn.functional as F


import lightning as L

import glob

from network_genes import Network as Network_g
from network_genes import ContrastiveLearning
from network_smri import Network as Network_smmri

from torchmetrics import Accuracy
from torchmetrics.classification import BinaryRecall,BinarySpecificity

from torch.optim.lr_scheduler import MultiStepLR
# %%
class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(in_features=612,out_features=250)
        self.second_layer = nn.Linear(in_features=250, out_features=200)
        self.third_layer = nn.Linear(in_features=200,out_features=150)
        self.layer4 = nn.Linear(150,2)
        dropout = nn.Dropout(0.25)
        self.all_layers = nn.Sequential(self.first_layer,nn.ReLU6(),dropout,
                                    self.second_layer,nn.ReLU6(),dropout, 
                                    self.third_layer,nn.ReLU6(),dropout,
                                    self.layer4)

    def forward(self, gene,data=None):
        # self.first_layer.weigh
        img = self.all_layers(gene)
        return img
    
# %%
class ContrastiveClassification(L.LightningModule):
    def __init__(self, network=None,
                 load_from:str='5156841',
                 learning_rate = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['network','contrastive_model'])
        self.model = network
        self.lr = learning_rate
        check_point = glob.glob(f'./tb_logs/{load_from}/version_0/checkpoints/*.ckpt')[0]
        self.contrastive_model = ContrastiveLearning.load_from_checkpoint(check_point
                                                     ,smri_network=Network_smmri()
                                                     ,gene_network=Network_g())
        self.contrastive_model.requires_grad_(False)
        self.accuracy = Accuracy(task = 'binary')
        self.recall = BinaryRecall()
        self.speficity = BinarySpecificity()

        self.test_outputs = []

    def forward(self, X_genes,X_smri):
       with torch.no_grad():
            pred_genes = self.contrastive_model.gene_NN(X_genes.float())
       input = torch.cat([X_smri.float(),pred_genes],axis=1)
       pred = self.model(input)
       x = F.log_softmax(pred,dim=1)
       return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X_genes,X_smri,y,_ = batch
        logits = self(X_genes,X_smri)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds,y)
        specificity = self.speficity(preds,y)
        values = {
            'train_loss' : loss,
            'train_acc' : (recall + specificity)/2
        }
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)
        

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X_genes,X_smri,y,_ = batch
        logits = self(X_genes,X_smri)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds,y)
        specificity = self.speficity(preds,y)
        values = {
            'valid_loss' : loss,
            'valid_acc' : (recall + specificity)/2
        }
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx,dataloader_idx):

        X_genes,X_smri,y,_ = batch
        logits = self(X_genes,X_smri)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds,y)
        specificity = self.speficity(preds,y)
        values = {
            'test_loss' : loss,
            'test_acc' : (recall + specificity)/2
        }
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def on_test_epoch_end(self) -> None:
        return super().on_test_epoch_end()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.lr,
                                    weight_decay=1e-5
                                    )
        lr_scheduler = MultiStepLR(optimizer, milestones=[15,20,30,35,40], gamma=0.1)
        return [optimizer],[lr_scheduler]
