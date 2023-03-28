import torch
from utils.nn.model.ParticleNetGlobal import ParticleNetTagger
from torch import Tensor
import math

def get_model(data_config, **kwargs):
    conv_params = [
        #(16, (32, 32, 32)),
        #(16, (64, 64, 64)),
        #(16, (128, 128, 128)),
        #(16, (256, 256, 256)),
        ]
    fc_params = []#[(256, 0.01)]
    fc_global_params = []#[(256,0.01)] 
    fc_combined_params = [(5,0.0)]# (256, 0.1)]#, (256, 0.1)] 
    use_fusion = True
    batch_norm = True #EdgeConv and feature conv batch norm
    global_batch_norm = False
    conv_dim = 12 # dimension of feature convolution layer (default:32)
    eflow_features_dims  = len(data_config.input_dicts['eflow_features'])
    global_features_dims = len(data_config.input_dicts['global_features'])
    # training linear and quadratic together:
    num_classes = 2 #len(data_config.label_value)
    model = ParticleNetTagger(eflow_features_dims, global_features_dims, num_classes,
                              conv_params, 
                              fc_params=fc_params, fc_global_params=fc_global_params, fc_combined_params=fc_combined_params,
                              use_fusion=use_fusion,
                              batch_norm=batch_norm,
                              global_batch_norm=global_batch_norm,
                              conv_dim=conv_dim,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              constituents_input_dropout=kwargs.get('constituents_input_dropout', None),
                              for_inference=False,
                              )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
        }

    return model, model_info

class LossLikelihoodFree(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'sum') -> None:
        super(LossLikelihoodFree, self).__init__(None, None, reduction)

        self.base_points = [.5, 1, 1.5, 2, 2.5, 5] # Could add to data_config, but the Loss function is not called with the data_config, so hard-code for now

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        # fit the linear term only for now (the network has just one output anyways)
        weight      = target[:,0] # this is the weight
        target_lin  = target[:,1] # this is the linear term
        target_quad = target[:,2] # this is the quadratic term

        if torch.isnan(target).sum():
            print("Found NAN in target!")
        if torch.isnan(input).sum():
            print("Found NAN in input!")
        #print(input)

        loss = weight*( self.base_points[0]*(target_lin-input[:,0]) + self.base_points[0]**2*(target_quad-input[:,1]) )**2
        for theta_base in self.base_points[1:]:
            loss += weight*( theta_base*(target_lin-input[:,0]) + theta_base**2*(target_quad-input[:,1]) )**2

        #for theta_base in self.base_points[1:]: #two base-points: 1,2
        #    loss += weight*( theta_base*(target_lin-input[:,0]) + .5*theta_base**2*(target_quad-input[:,1]) )**2

        #print ("weight") 
        #print (weight) 
        #print ("target_") 
        #print (target_) 
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

