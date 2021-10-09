
args = {
  'dataset_type': 'images', # images, speech , tabuler
  'algorithm_type':'simclr',
  'input_mix':False,
  'low_resolution':False,
  'pretrained':False,
  #mixup coeffecient
  'alpha':2,
  #Temperature of contrastice loss
  't':0.2,

  'batch_size':512,
  'num_workers':1,
   
}