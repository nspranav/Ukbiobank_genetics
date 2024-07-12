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

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Linear(in_features,out_features,**factory_kwargs)
        torch.nn.init.xavier_uniform_(self.layer.weight,gain=5.0)

        


    def forward(self, x):
        return self.layer(x)
    
    def __repr__(self):
        return super().__repr__()

    # def reset_parameters(self) -> None:
    #     # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    #     # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    #     # https://github.com/pytorch/pytorch/issues/57109
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         init.uniform_(self.bias, -bound, bound)


# %%
class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(in_features=512,out_features=250)
        self.second_layer = nn.Linear(in_features=250, out_features=200)
        self.third_layer = nn.Linear(in_features=200,out_features=150)
        self.layer4 = nn.Linear(in_features=150, out_features= 100)
        self.layer5 = nn.Linear(100,32)
        dropout = nn.Dropout(0.25)
        self.all_layers = nn.Sequential(self.first_layer,nn.ReLU(),dropout,
                                    self.second_layer,nn.ReLU(), 
                                    self.third_layer,nn.ReLU(),
                                    self.layer4)
        

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
