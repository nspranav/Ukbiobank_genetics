# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

#%%
class CustomLayer(nn.Module):
    """
    Custom layer created to accomodate sparse connections
    """
    def __init__(self, num_layer: int = 1, in_features: int = 1060, 
                 out_features: int = 149) -> None:
        super(CustomLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Linear(in_features,out_features)
        torch.nn.init.xavier_uniform_(self.layer.weight,gain=15.0)

        self.num_layer = num_layer
        if self.num_layer == 1:
            self.adj_matrix = pd.read_csv('adj_matrix_first_second.csv',
                                          index_col = 'SNP_id').T
        else:
            self.adj_matrix = pd.read_csv('adj_matrix_second_third.csv',
                                          index_col = 'converted_alias').T
        self.adj_matrix = self.adj_matrix.fillna(0)


    def forward(self, x):

        #device = "cpu"
        # self.layer.weight = nn.Parameter(self.layer.weight *  
        #                                  torch.Tensor(self.adj_matrix
        #                                               .to_numpy()).to(device))
        return self.layer(x)

    def __repr__(self):
        return f'The weight matrix shape: {self.layer.weight.shape}, the adj_matrix shape: {self.adj_matrix.shape}'



# %%
class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1060)
        self.first_layer = CustomLayer(num_layer=1,in_features=1060,out_features=149)
        self.bn1 = nn.BatchNorm1d(1)
        self.second_layer = CustomLayer(num_layer=2, in_features=149, out_features=1283)
        self.bn2 = nn.BatchNorm1d(1)
        self.third_layer = nn.Linear(1283,2)
        self.all_layers = nn.Sequential(self.first_layer,nn.Tanh(),
                                    self.second_layer,nn.Tanh(),
                                    self.third_layer)
        

    def forward(self, gene,data=None):
        # self.first_layer.weigh
        img = self.all_layers(gene)

        return img
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m,nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    




# %%
