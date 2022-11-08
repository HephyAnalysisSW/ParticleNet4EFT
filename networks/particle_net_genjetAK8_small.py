import torch
from utils.nn.model.ParticleNet import ParticleNetTagger
from torch import Tensor
import math

def get_model(data_config, **kwargs):
    conv_params = [
        (4, (16, 16, 16)),
        (4, (32, 32, 32)),
        (4, (64, 64, 64)),
        ]
    fc_params = [(64, 0.1)]
    use_fusion = True

    chh_features_dims = len(data_config.input_dicts['chh_features'])
    neh_features_dims = len(data_config.input_dicts['neh_features'])
    el_features_dims = len(data_config.input_dicts['el_features'])
    mu_features_dims = len(data_config.input_dicts['mu_features'])
    ph_features_dims = len(data_config.input_dicts['ph_features'])
    num_classes = len(data_config.label_value)
    model = ParticleNetTagger(chh_features_dims, neh_features_dims, el_features_dims, mu_features_dims, ph_features_dims, num_classes,
                              conv_params, fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              chh_input_dropout=kwargs.get('chh_input_dropout', None),
			      neh_input_dropout=kwargs.get('neh_input_dropout', None),
                              el_input_dropout=kwargs.get('el_input_dropout', None),
			      mu_input_dropout=kwargs.get('mu_input_dropout', None),
			      ph_input_dropout=kwargs.get('ph_input_dropout', None),
                              for_inference=kwargs.get('for_inference', False)
                              )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        #'output_names':['softmax'],
        #'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
        }

    return model, model_info

class LogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(LogCoshLoss, self).__init__(None, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input - target
        loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss(data_config, **kwargs):
    #return LogCoshLoss()
    return torch.nn.MSELoss() 

