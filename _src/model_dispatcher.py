from _src.models import TabulerModel
'''
This modules responsible of creating models
'''
def get_model(args, model_type: str = 'mlp'):
  model = None
  if model_type == 'mlp':
    model = TabulerModel(
      args.hidden_dim,
       args.input_dim,
        args.head_dim, 
        args.num_classes, 
        args.pool_size
        )
  return model


    
