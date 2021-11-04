import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import os

class DatasetTabular(Dataset):
  '''
  Tabular data class
  '''
  def __init__(self, data, y ):
    super().__init__()
    self.data=data.values
    self.y=y.values
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    return self.data[idx],self.y[idx]



def _create_dataset(path):
  '''
  create dataset object from path
  '''
  df=pd.read_csv(path,header=None)
  df.iloc[:,-1]=df.iloc[:,-1]-1
  df.iloc[:,:11]=(df.iloc[:,:11]-df.iloc[:,:11].mean())/df.iloc[:,:11].std()
  x,y=df.iloc[:,:-1],df.iloc[:,-1]
  dataset=DatasetTabular(x,y)
  return dataset

def generate_splits(path, split):
  '''
  split data 
  '''
  dataset = _create_dataset(path)
  train_sz, test_sz=len(dataset)-int(len(dataset)*split),int(len(dataset)*split)
  train,test=torch.utils.data.random_split(dataset,[train_sz ,test_sz])
  return train, test


def tabularaugment(data,prob=0.2):
  '''
  Data Augmentation adding noise
  '''
  matrix=torch.ones(data.shape)*prob
  
  p=torch.bernoulli(matrix).to(data.device)
 
  output =data+p*torch.randn(data.shape,device=data.device)

  return output 

