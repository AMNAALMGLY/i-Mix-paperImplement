from torch import nn as nn
from torch.distributions import Beta
import torch
class Mixup:
  def __init__(self,alpha):
    '''
    mixup the data with mixing coeffecient
    '''
    super().__init__()
    self.alpha=alpha
  def __call__(self,x):
    lam=Beta(self.alpha,self.alpha).sample()
    randidx = torch.randperm(len(x)).to(x.device)
    x = lam * x + (1-lam) * x[randidx]
    return x, lam , randidx


class Maxout(nn.Module):
    '''
    Apply a maxout pooling layer
    '''
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m

def seed_everything(seed):
  print("seeding..is not implemented yet...")