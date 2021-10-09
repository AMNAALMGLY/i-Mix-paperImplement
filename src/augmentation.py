
import torch

def tabularaugment(input,prob=0.2):
  '''
  Data Augmentation adding noise
  '''
  matrix=torch.ones(input.shape)*prob
  
  p=torch.bernoulli(matrix).to(input.device)
 
  output =input+p*torch.randn(input.shape,device=input.device)

  return output 
