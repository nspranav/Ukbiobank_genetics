# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import math
from torch.nn import init
from torch.nn.parameter import Parameter

#%%
class CustomLayer(nn.Module):
    """
    Custom layer created to accomodate sparse connections
    """
    def __init__(self, num_layer: int = 1, in_features: int = 1060, 
                 out_features: int = 149,device=None, dtype=None) -> None:
        # super(CustomLayer, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features

        # self.layer = nn.Linear(in_features,out_features)
        # torch.nn.init.xavier_uniform_(self.layer.weight,gain=15.0)

        # self.num_layer = num_layer
        # if self.num_layer < 3:
        #     if self.num_layer == 1:
        #         self.adj_matrix = pd.read_csv('adj_matrix_first_second.csv',
        #                                     index_col = 'SNP_id').T
        #     elif self.num_layer == 2:
        #         self.adj_matrix = pd.read_csv('adj_matrix_second_third.csv',
        #                                     index_col = 'converted_alias').T
            

        #     self.adj_matrix = self.adj_matrix.fillna(0)
        #     self.adj_matrix[self.adj_matrix == 0] = 1

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))

        #self.reset_parameters()

        # initialize weights and biases
        init.xavier_uniform_(self.weight,gain=1.0) # weight init

        # self.num_layer = num_layer
        # if self.num_layer < 3:
        #     if self.num_layer == 1:
        #         self.adj_matrix = pd.read_csv('adj_matrix_first_second.csv',
        #                                     index_col = 'SNP_id').T
        #     elif num_layer == 2:
        #         self.adj_matrix = pd.read_csv('adj_matrix_second_third.csv',
        #                                     index_col = 'converted_alias').T
            

        #     self.adj_matrix = self.adj_matrix.fillna(0)
        #     #self.adj_matrix[self.adj_matrix == 0] = 1
        


    def forward(self, x):
        # with torch.no_grad():
        #     #print(f'shape = {self.weights.shape}, {self.adj_matrix.shape}')
        #     self.weight[self.adj_matrix.to_numpy() == 0] == 0
                
        return F.linear(x, self.weight, self.bias)
    
    def __repr__(self):
        # if self.num_layer < 3:
        #     return f'The weight matrix shape: {self.weights.shape}, the adj_matrix shape: {self.adj_matrix.shape}'
        # else:
        return super().__repr__()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


# %%
class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1060)
        self.first_layer = CustomLayer(num_layer=1,in_features=912,out_features=500)
        self.bn1 = nn.BatchNorm1d(1)
        self.second_layer = CustomLayer(num_layer=2, in_features=500, out_features=250)
        self.bn2 = nn.BatchNorm1d(1)
        self.third_layer = nn.Linear(in_features=250,out_features=2)
        self.layer4 = nn.Linear(in_features=400, out_features= 200)
        self.layer5 = nn.Linear(in_features=200,out_features=2) 

        self.all_layers = nn.Sequential(self.first_layer,nn.ReLU6(),
                                    self.second_layer,nn.ReLU6(),
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
