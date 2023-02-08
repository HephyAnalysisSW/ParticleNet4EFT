import torch
from torch import nn
from torch import Tensor

# create a MLP class
class MLP(nn.Module):

    def __init__(self, input_nfeatures, num_classes, hidden_layers=(), **kwargs):
        super().__init__(**kwargs)
        channels = [input_nfeatures] + list(hidden_layers) + [num_classes]
        layers = []
        for c, c_next in zip(channels[:-1], channels[1:]):
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.Linear(c, c_next, bias=True))
            # layers.append(nn.BatchNorm1d(c_next))
            layers.append(nn.ReLU())
        del layers[-1:]
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.mlp(x)


def get_model(data_config, **kwargs):
    hidden_layers = (300,100,100)
    _, pf_length, pf_feature_dims = data_config.input_shapes['hl_features']
    # print(pf_length, pf_feature_dims)
    input_dims = pf_length * pf_feature_dims
    num_classes = 2 #len(data_config.label_value)
    model = MLP(input_dims, num_classes, hidden_layers=hidden_layers)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
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