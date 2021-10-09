import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.trainer import Npair, TestSUP
from src.config import args
import copy

def setup_experiment(model, 
                     train_loader, validation_loader, test_loader,
                     experiment, pretrained_checkpoint, args):
    
    seed_everything(args['seed'])
    