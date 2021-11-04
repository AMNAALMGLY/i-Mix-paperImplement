import torch
from torch import nn
from _src.utils import Maxout, Mixup

class TabulerModel(nn.Module):
    '''
    5 layer Mlp with Projection Head and Maxout layer
    '''
    def __init__(self,hid_dim,input_dim,head_dim,num_classes,pool_size):
        super(TabulerModel, self).__init__()
        
        self.layer = nn.Sequential(nn.Linear(input_dim,hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim,hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim,hid_dim*2),
                    nn.BatchNorm1d(hid_dim*2),
                    nn.ReLU(),
                    nn.Linear(hid_dim*2,hid_dim*2),
                    nn.BatchNorm1d(hid_dim*2),
                    nn.ReLU(),
                    nn.Linear(hid_dim*2,hid_dim*2*2),
                    nn.BatchNorm1d(hid_dim*2*2)) 
        self.maxout=Maxout(pool_size)
        self.projectHead = nn.Sequential(nn.Linear(hid_dim,hid_dim),
                                         nn.ReLU(),
                                         nn.Linear(hid_dim,head_dim))
        
       
        
    def forward(self, inputs,):
        x = self.layer(inputs)
        x_max = self.maxout(x)
        
        return self.projectHead(x_max)

