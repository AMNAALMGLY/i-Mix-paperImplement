import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import args
from torchaudio.datasets import SPEECHCOMMANDS
import os
# tabular

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

# images
def get_image_CIFAR10(savedpath,transform,train=True):
  return torchvision.datasets.CIFAR10(savedpath,transform=transform,download=True,train=train) 

def get_image_CIFAR100(savedpath,transform,train=True):
  return torchvision.datasets.CIFAR100(savedpath,transform=transform,download=True,train=train) 
# speech
class SpeechData(SPEECHCOMMANDS):
    def __init__(self,subset: str = None, transform=None,silencePercent=0.1):
       
        super().__init__(f"./", download=True)
        self.CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')

        self.class_to_idx={c:i for i,c in enumerate(self.CLASSES)}
        def load_list(filename):
            i=0
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                for line in fileobj:
                    
                    return [os.path.join(self._path, line.strip())]               

        if subset == "validation":
          
            self._walker = load_list("validation_list.txt")
            print(self._walker)
       
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)         
            self._walker = [w for w in self._walker if w not in excludes]
            
     
        self.transform=transform
        #add silence
        self._walker+=[f'./SpeechCommands/speech_commands_v0.02/'+'silence'+'' * i for i in range(int(len(self._walker)*silencePercent))]
        if self.transform:
          self.transform(self._walker)
    def __getitem__(self, index):
      '''
      return .wav file path and the target if it is in classes 
      '''
  
      label= self._walker[index].split('/')[3]
   
      if label not in CLASSES:
        label='unknown'
      return self._walker[index],self.class_to_idx[label]



# Dataloaders
def get_loaders(experiment,dataset_type,data_path=None,val_split=None,val_sz=None,train_sz=None,imagedataset=None,transform=None):
  '''
  depending on the experiment[pretrain, finetune] & dataset type (tablular, speech, images)
  return train, val and test loaders according to val_split
  images could be cifar10 or cifar100
  tranformation applied to the data is passed as transform arg
  '''
  pin_memory = True if torch.cuda.is_available() else False
  
  if dataset_type=='image':
    #Generate train, val, test dataset 
    
    if imagedataset=='cifar10':
      train_1=get_image_CIFAR10(data_path,transform)
      train_sz, val_sz=len(train_1)-int(len(train_1)*val_split),int(len(train_1)*val_split)
      train,val=torch.utils.data.random_split(train_1,[train_sz ,val_sz])
   
      test=get_image_CIFAR10(data_path,transform,train=False,)
    elif imagedataset=='cifar100':

      train_1=get_image_CIFAR100(data_path,transform)
      train_sz, val_sz=len(train_1)-int(len(train_1)*val_split),int(len(train_1)*val_split)
      train,val=torch.utils.data.random_split(train1,[train_sz ,val_sz])
    
      test=get_image_CIFAR100(data_path,transform,train=False,)
    #generate dataloader
    train1Loader=DataLoader(train_1, batch_size=args['batch_size'], 
                             num_workers=args['num_workers'], 
                            pin_memory=pin_memory)
    trainLoader=DataLoader(train, batch_size=args['batch_size'], 
                             num_workers=args['num_workers'], 
                            pin_memory=pin_memory)
    valLoader= DataLoader(val, batch_size=args['batch_size'], 
                             num_workers=args['num_workers'], 
                            pin_memory=pin_memory) 
    testLoader=DataLoader(test, batch_size=args['batch_size'], 
                             num_workers=args['num_workers'], 
                            pin_memory=pin_memory)  
    if experiment=='pretrain':
      return  train1Loader ,testLoader
    else:
      return   trainLoader,  valLoader,  testLoader                                     
  elif dataset_type=='speech':
    '''
    helper functions for the dataloaders
    '''
    def pad_sequence(batch):
          batch=[b.t()for b in batch]
          batch=nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
          return batch.permute(0,2,1)
    def collate_fn(batch):
          tensors=[]
          targets=[]
          for path,target in batch:
            tensor,_=torchaudio.load(path)
            tensors+=[tensor]
            targets+=[target]
          tensors=pad_sequence(tensors)
          targets=torch.stack(targets)
          return tensors,targets
    
    
    train=SpeechData('training')
    train,val,test=torch.utils.data.random_split(train,[train_sz ,val_sz,val_sz])
    #train for the pretrain
    train2=torch.cat((train,val),dim=0)
    
    trainLoader = torch.utils.data.DataLoader(
        train,
        batch_size=args['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args['num_workers'],
        pin_memory=pin_memory,
    )
    train1Loader = torch.utils.data.DataLoader(
        train2,
        batch_size=args['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args['num_workers'],
        pin_memory=pin_memory,
    )
    testLoader = torch.utils.data.DataLoader(
        test,
        batch_size=args['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args['num_workers'],
        pin_memory=pin_memory,
    )
    valLoader = torch.utils.data.DataLoader(
        val,
        batch_size=args['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args['num_workers'],
        pin_memory=pin_memory,
    )
    if experiment=='pretrain':
      return  train1Loader ,testLoader
    else:
      return   trainLoader,  valLoader,  testLoader

#testig image loaders
'''
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
    
transform=   transforms.Compose([
      transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
image_loaderTrain,_=get_loaders(experiment='pretrain',dataset_type='image',data_path='./data',val_split=0.2,imagedataset='cifar10',transform=transform)
for i , j in     image_loaderTrain:
  print(i,j)
  break
''' 
#testing speech loaders
speech_loaderTrain,_=get_loaders(experiment='pretrain',dataset_type='speech',data_path='./data',val_sz=7000,train_sz=51000)
for i , j in    speech_loaderTrain:
  print(i,j)
  break
 



