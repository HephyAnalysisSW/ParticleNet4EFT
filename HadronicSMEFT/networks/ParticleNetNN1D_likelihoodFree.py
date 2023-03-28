import torch
from utils.nn.model.ParticleNetNN import ParticleNetTagger
from torch import Tensor
import math

def get_model(data_config, **kwargs):
    #params = [(256, 0.1), (256, 0.1)]#, (256, 0.1)] 
    params = [(5, 0.0)]#, (256, 0.1)]#, (256, 0.1)] 
    batch_norm = False
    dims = len(data_config.input_dicts['features'])
    # training linear and quadratic together:
    num_classes = 1 #len(data_config.label_value)
    model = ParticleNetTagger(dims, params, num_classes,
                              batch_norm=batch_norm,
                              )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        }

    print ("model_info",model_info)

    return model, model_info

class LossLikelihoodFree(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'sum') -> None:
        super(LossLikelihoodFree, self).__init__(None, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        # fit the linear term only for now (the network has just one output anyways)

        #print (target.shape, target)
        weight      = target[:,0] # this is the weight
        target_     = target[:,1] # this is the linear term

        if torch.isnan(target_).sum():
            print("Found NAN in target!")
        if torch.isnan(input).sum():
            print("Found NAN in input!")
        #print(input)

        loss = weight*( (target_-input) )**2

        #print ("weight") 
        #print (weight) 
        #print ("target_lin") 
        #print (target_lin) 
        #print ("output_lin") 
        #print (input[:,0]) 
        #print ("target_quad") 
        #print (target_quad) 
        #print ("output_quad") 
        #print (input[:,1]) 
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

