import torch
from utils.nn.model.ParticleNetGlobal import ParticleNetTagger
from torch import Tensor
import math

def get_model(data_config, **kwargs):
    conv_params =  [(3, (300,300,300))] # Note to myself: first parameter in conv_params is (probably) the number of connected neighbours
    fc_params = [(15, 0.0), (2, 0.0)]           # fully connected layer (hidden_size, drop_out) after GNN for array based input
    fc_global_params = [(80, 0.0),(12, 0.0)]   # fully connected layer (hidden_size, drop_out) for global features only
    fc_combined_params = [(5,0.0)] # fully connected layer (hidden_size, drop_out) for combined input from fc and fc_global
    use_fusion = False
    
    eflow_features_dims    = len(data_config.input_dicts['eflow_features'])
    global_features_dims = len(data_config.input_dicts['global_features'])
    # training linear and quadratic together:
    num_classes = 2 
    model = ParticleNetTagger(eflow_features_dims, global_features_dims, num_classes,
                              conv_params, fc_params, fc_global_params, fc_combined_params, 
                              use_fusion=use_fusion, # applies batchnorm1D (?)
                              use_fts_bn=kwargs.get('use_fts_bn', False), # ?
                              use_counts=kwargs.get('use_counts', False), # ?
                              constituents_input_dropout=kwargs.get('constituents_input_dropout', None), # drop out before ParticleNet
                              global_input_dropout=kwargs.get('global_input_dropout', None),  # drop out before first DNN layer
                              for_inference=kwargs.get('for_inference', False) # applies softmax at last step
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

class LossLikelihoodFree(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(LossLikelihoodFree, self).__init__(None, None, reduction)

        self.base_points = [1,2] # Could add to data_config, but the Loss function is not called with the data_config, so hard-code for now

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight      = target[:,0] # this is the weight
        target_lin  = target[:,1] # this is the linear term
        target_quad = target[:,2] # this is the quadratic term

        loss = weight*( self.base_points[0]*(target_lin-input[:,0]) + self.base_points[0]**2*(target_quad-input[:,1]) )**2
        for theta_base in self.base_points[1:]: #two base-points: 1,2
            loss += weight*( theta_base*(target_lin-input[:,0]) + theta_base**2*(target_quad-input[:,1]) )**2 #removed the 0.5 


        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss(data_config, **kwargs):
    return LossLikelihoodFree()
    #return torch.nn.MSELoss() 

