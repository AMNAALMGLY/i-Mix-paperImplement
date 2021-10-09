import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta
import pytorch_lightning as pl
from torchmetrics import Accuracy
from  pl_bolts import optimizers



class Npair(pl.LightningModule):
  '''
  pretrain an N-Pair model with i-mix Loss
  '''
 
  def __init__(self,model,alpha, augment,t,learningRate,weight_decay, momentum, use_imix=False):
     super().__init__()
     self.model=model
     self.alpha=alpha
     self.mixup=Mixup(alpha)
     self.augment=augment
     self.t=t
     self.learningRate=learningRate
     self.weight_decay=weight_decay
     self.momentum=momentum
     self.setup_criterion()
     self.use_imix=use_imix
     
  def _shared_step(self,batch,batch_idx):
    x,_=batch
    
    r= self.augment(x.float())
    
    r_prime=self.augment(x.float())
    #calculating the loss
    if self.use_imix:
      r_mix , lam , randidx=self.mixup(r)

      randidx = randidx.to(self.device)
      logits = torch.matmul(F.normalize(model(r_mix)), F.normalize(model(r_prime)).T) / self.t
      loss= lam * self.criterion(logits, torch.arange(len(x)).to(self.device)) + \
        (1-lam) * self.criterion(logits, randidx)
     
    else:
      logits=torch.matmul(F.normalize(model(r)), F.normalize(model(r_prime)).T) / self.t
      loss=self.criterion(logits, torch.arange(len(x)).to(x.device))
      
    return loss


  def training_step(self, batch,batch_idx):
    loss=self._shared_step(batch,'train')
    print(loss)
    self.log('loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
    return {'loss': loss}
  def validation_step(self,batch,batch_idx):
    loss=self._shared_step(batch,'val')
 
    self.log('val_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
    
    return {'val_loss': loss}
  def configure_optimizers(self): 
    optimizer =torch.optim.SGD(self.parameters(), lr=self.learningRate,weight_decay=self.weight_decay,momentum=self.momentum)
    scheduler=optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=10, max_epochs=5000)
    return {
                'optimizer': optimizer,
                "lr_scheduler": scheduler
            }



  def setup_criterion(self):
    self.criterion= nn.CrossEntropyLoss()



class TestSUP(pl.LightningModule):
  '''
  create a fine tuned model
  TODO:Freeze_layers
  '''
 
  def __init__(self,model,fc,learningRate,weight_decay, momentum,num_output,freeze_encoder=True,layers_to_freeze=6):
     super().__init__()
     self.model=model
     self.fc=fc
     self.learningRate=learningRate
     self.weight_decay=weight_decay
     self.momentum=momentum
     self.setup_criterion()
     self.train_acc=Accuracy(num_classes=num_output)
     self.validation_acc=Accuracy(num_classes=num_output)
     self.test_acc=Accuracy(num_classes=num_output)
     if freeze_encoder:
       self.model.eval()
       ct=0
       for child in self.model.children():
            ct += 1
            if ct < layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
       #self.model.projectHead=fc
  def forward(self,x):
    return self.model(x)
     

  def _shared_step(self, batch, batch_idx,accuracy):
    x,y=batch
    pretrain=self.model(x.float())
   
    #output=self.fc(pretrain)
    prediction=torch.sigmoid(pretrain)
    loss=self.criterion(pretrain.squeeze(),y)
  
    accuracy.update(prediction,y)
    return loss 

  def training_step(self, batch,batch_idx):
    
    loss=self._shared_step(batch,batch_idx,self.train_acc)
    self.log('train_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
  
    return {'loss': loss}
  def train_epoch_end(self, out): 
    self.log('training accuracy',self.train_acc.compute(),prog_bar=True,)
    self.train_acc.reset()
  def validation_step(self, batch,batch_idx):
    loss=self._shared_step(batch,batch_idx,self.validation_acc)
 
    self.log('val_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
  
 
  def validation_epoch_end(self, out):
    self.log('validation accuracy',self.validation_acc.compute(),prog_bar=True,)
    print(self.validation_acc.compute())
    self.validation_acc.reset()
  
  def test_step(self,batch,batch_idx):
     loss=self._shared_step(batch,batch_idx,self.test_acc)
     self.log('test_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
  def test_epoch_end(self, out): 
    self.log('test accuracy',self.test_acc.compute(),prog_bar=True,)
    self.test_acc.reset()

  def configure_optimizers(self): 
    optimizer =torch.optim.SGD(self.parameters(), lr=self.learningRate)
    scheduler=optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=10, max_epochs=500)
    return {
                'optimizer': optimizer,
                "lr_scheduler": scheduler
            }

  def setup_criterion(self):
    self.criterion= nn.CrossEntropyLoss()

 

