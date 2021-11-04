import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from  pl_bolts import optimizers
from torch.utils.data import Dataset, DataLoader

from _src.model_dispatcher import get_model
from _src.utils import seed_everything, Mixup
from _src.baselines import Npair
from _src.dataset import tabularaugment, generate_splits

import argparse

pretrained_dir = './checkpoints/pretrained/'
finetuned_dir = './checkpoints/finetuned/'
data_path = './data/covtype.data'

def _build_model(args):
  model = None
  if args.dataset_type == 'covtype':
    model = get_model(args,'mlp')
  return model

def _load_model(args):
  model = _build_model(args)
  # is it the same to laod model as in normal pytorch

def _pretrain(args, model, train_loader):
  pretrained_checkpoint_file = args.pretrained_checkpoint_file
  checkpoint_model_path = os.path.join(pretrained_dir, pretrained_checkpoint_file)
  LitModel=Npair(
              model,
              alpha=args.alpha,
              augment = tabularaugment,
              t= args.t,
              learningRate= args.lr,
              weight_decay= args.weight_decay,
              momentum=args.momentum,
              use_imix=args.use_imix
              )
  checkpoint_callback = ModelCheckpoint(monitor="val_loss",mode='min',
                                      filename="i-Mix-fineT-{epoch:02d}-{val-loss:.2f}",
                                      dirpath= pretrained_dir,
                                      verbose=True,
                                      save_last=True)
                                      
  logger = TensorBoardLogger("tb5_logs", name="LitModel")

  trainer = None
  if args.resume:
    trainer=pl.Trainer(max_epochs=400, gpus=-1,auto_select_gpus=True,
                    callbacks=[checkpoint_callback],
                    resume_from_checkpoint=checkpoint_model_path,
                    logger=logger)
  else:
    trainer=pl.Trainer(max_epochs=400, gpus=-1,auto_select_gpus=True,
                    callbacks=[checkpoint_callback],
                    logger=logger)
  trainer.fit(LitModel,train_loader)


def main(args):
  seed_everything(args.seed)
  if args.pretrain:
    # setup pretraining modules
    print("="*50)
    print("... Start Pre-Training ...")
    print("="*50)

    
    train, test=generate_splits(data_path,split=0.99)
    #create pretrain dataloader
    pin_memory = True if torch.cuda.is_available() else False
    train_loader=DataLoader(train,batch_size=args.batch_size,pin_memory=pin_memory)
    test_dataloader=DataLoader(test,batch_size=args.batch_size,pin_memory=pin_memory)


    model = _build_model(args)
    model = _pretrain(args, model, train_loader)
    pass
  else:
    # setup finetuning modules
    model = _load_model(args,)
    print("="*50)
    print("... Start Fine-Tuning ...")
    print("="*50)
    model = _fine_tune(model, args)
    pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--pretrained_checkpoint_file', type=str, default='')
  parser.add_argument('--dataset_type', type=str, default='covtype')


  parser.add_argument('--hidden_dim', type=int, default=2048)
  parser.add_argument('--input_dim', type=int, default=54)
  parser.add_argument('--head_dim', type=int, default=128)
  parser.add_argument('--num_classes', type=int, default=7)
  parser.add_argument('--pool_size', type=int, default=4)
  parser.add_argument('--num_epochs', type=int, default=400)
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--batch_size', type=int, default=512)
  parser.add_argument('--alpha', type=int, default=2)

  parser.add_argument('--lr', type=float, default=.125)
  parser.add_argument('--t', type=float, default= .2)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--momentum', type=float, default=.99)

  parser.add_argument('--freeze_encoder', type=bool, default=True)
  parser.add_argument('--pretrain', type=bool, default=True)
  parser.add_argument('--use_imix', type=bool, default=True)
  parser.add_argument('--resume', type=bool, default=False)

  args = parser.parse_args()

  main(args)