import torch
from .PNGlobal_TopDecay import ParticleNetEFT
from torch import Tensor
import math

def get_model(data_config, **kwargs):
    conv_params = [(4, (8, 8, 8)), (4, (16, 16, 16))]
    pnet_fc_params = [(32, 0.1)]
    freeze_pnet = False
    globals_fc_params = [(300, 0.1), (300, 0.1), (100, 0.1)]
    freeze_global_fc = False
    joined_fc_params = [(64, 0.1)]
    use_fusion = True

    eflow_features_dims    = len(data_config.input_dicts['eflow_features'])
    global_features_dims = len(data_config.input_dicts['global_features'])
    print(f"{eflow_features_dims=}")
    print(f"{global_features_dims=}")

    # training linear and quadratic together:
    num_classes = 2 #len(data_config.label_value)
    model = ParticleNetEFT(eflow_features_dims,
                           global_features_dims,
                           num_classes,
                           feature_conv_out_dim=32,
                           conv_params=kwargs.get("conv_params", conv_params),
                           pnet_fc_params=kwargs.get("pnet_fc_params", pnet_fc_params),
                           freeze_pnet=kwargs.get('freeze_pnet', freeze_pnet),
                           globals_fc_params=kwargs.get("globals_fc_params",globals_fc_params),
                           freeze_global_fc=kwargs.get('freeze_global_fc', freeze_global_fc),
                           joined_fc_params=kwargs.get("joined_fc_params", joined_fc_params),
                           use_fusion=kwargs.get("use_fusion", use_fusion),
                           use_fts_bn=kwargs.get('use_fts_bn', True),
                           use_counts=kwargs.get('use_counts', True),
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
        #print ("input")
        #print (input)
        #print ("target")
        #print (target)
        # fit the linear term only for now (the network has just one output anyways)
        weight      = target[:,0] # this is the weight
        target_lin  = target[:,1] # this is the linear term
        target_quad = target[:,2] # this is the quadratic term

        loss = weight*( self.base_points[0]*(target_lin-input[:,0]) + .5*self.base_points[0]**2*(target_quad-input[:,1]) )**2
        for theta_base in self.base_points[1:]: #two base-points: 1,2
            loss += weight*( theta_base*(target_lin-input[:,0]) + .5*theta_base**2*(target_quad-input[:,1]) )**2

        #print ("weight") 
        #print (weight) 
        #print ("target_lin") 
        #print (target_lin) 
        #print ("loss") 
        #print (loss) 

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss(data_config, **kwargs):
    return LossLikelihoodFree()
    #return torch.nn.MSELoss() 

