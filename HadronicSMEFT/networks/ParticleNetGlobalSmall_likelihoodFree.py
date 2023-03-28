import torch
from utils.nn.model.ParticleNetGlobal import ParticleNetTagger
from torch import Tensor
import math

def get_model(data_config, **kwargs):
    conv_params = [
        (16, (3,)),
        #(16, (128, 128, 128)),
        #(16, (256, 256, 256)),
        ]
    fc_params = [(32, 0.)]
    fc_global_params = None 
    fc_combined_params = None#[(32, 0.),]# (256, 0.1)]#, (256, 0.1)] 
    use_fusion = False
    batch_norm = False #EdgeConv and feature conv batch norm
    conv_dim = 4 # dimension of feature convolution layer (default:32)
    eflow_features_dims  = len(data_config.input_dicts['eflow_features'])
    global_features_dims = len(data_config.input_dicts['global_features'])
    # training linear and quadratic together:
    num_classes = 2 #len(data_config.label_value)
    model = ParticleNetTagger(eflow_features_dims, global_features_dims, num_classes,
                              conv_params, 
                              fc_params=fc_params, fc_global_params=fc_global_params, fc_combined_params=fc_combined_params,
                              use_fusion=use_fusion,
                              batch_norm=batch_norm,
                              conv_dim=conv_dim,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              constituents_input_dropout=kwargs.get('constituents_input_dropout', None),
                              for_inference=kwargs.get('for_inference', False)
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

    def __init__(self, reduction: str = 'mean') -> None:
        super(LossLikelihoodFree, self).__init__(None, None, reduction)

        self.base_points = [1,2] # Could add to data_config, but the Loss function is not called with the data_config, so hard-code for now

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        # fit the linear term only for now (the network has just one output anyways)
        weight      = target[:,0] # this is the weight
        target_lin  = target[:,1] # this is the linear term
        target_quad = target[:,2] # this is the quadratic term

        ## make 2d plot

        #if True:
        #  with torch.no_grad():
        #    # 2D plots for convergence + animation
        #    import numpy as np
        #    import sys
        #    sys.path.insert(0, '..')
        #    from tools import helpers
        #    th2d = {}
        #    for i_der, der in enumerate( ["lin", "quad"] ):
        #        truth_ratio = target[:,i_der+1].numpy()/weight.numpy()
        #        quantiles = np.quantile(truth_ratio, q=(0.01,1-0.01))
        #        if len(der)==2: #quadratic
        #            binning = np.linspace( min([0, quantiles[0]]), quantiles[1], 21 )
        #        else:
        #            binning = np.linspace( quantiles[0], quantiles[1], 21 )
        #        print (der, "num events",len(truth_ratio))
        #        print ("truth", np.mean(truth_ratio))
        #        print ("pred ", np.mean(input[:,i_der].numpy()))
        #        print (binning)
        #        print ()
        #        th2d[der]      = helpers.make_TH2F( np.histogram2d( truth_ratio, input[:,i_der].numpy(), bins = [binning, binning], weights=weight.numpy()) )
        #        tex_name = "%s"%(",".join( der ))
        #        th2d[der].GetXaxis().SetTitle( tex_name + " truth" )
        #        th2d[der].GetYaxis().SetTitle( tex_name + " prediction" )

        if torch.isnan(target).sum():
            print("Found NAN in target!")
        if torch.isnan(input).sum():
            print("Found NAN in input!")
        #print(input)

        loss = weight*( self.base_points[0]*(target_lin-input[:,0]) + self.base_points[0]**2*(target_quad-input[:,1]) )**2
        for theta_base in self.base_points[1:]: #two base-points: 1,2
            loss += weight*( theta_base*(target_lin-input[:,0]) + theta_base**2*(target_quad-input[:,1]) )**2

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

