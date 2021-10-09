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
