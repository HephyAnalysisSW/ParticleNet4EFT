import numpy as np
import torch
import torch.nn as nn

class ParticleNet(nn.Module):

    def __init__(self,
                 dims,
                 params,
                 num_classes,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)
        fcs = []
        
        for idx, layer_param in enumerate(params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn_global = dims
            else:
                in_chn_global = params[idx - 1][0]

            fcs.append(nn.Sequential(nn.Linear(in_chn_global, channels), nn.ReLU(), nn.Dropout(drop_rate)))

        fcs.append(nn.Linear(params[-1][0], num_classes))
            
        self.fc = nn.Sequential(*fcs)

    def forward(self, features, mask=None):

        # global features
        for idx, layer in enumerate(self.fc): 
            if idx == 0:
                x = layer(features[:,:,0]) #evaluate global layers
            else:            
                x = layer(x)

        output = x

        return output

class ParticleNetTagger(nn.Module):

    def __init__(self,
                 dims,
                 params,
                 num_classes,
                 batch_norm=True,
                 #global_input_dropout=None,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.num_classes = num_classes
        #self.global_input_dropout = nn.Dropout(global_input_dropout) if global_input_dropout else None
        self.batchnorm = nn.BatchNorm1d(dims) if batch_norm else None
        self.pn = ParticleNet(dims=dims,
                              params=params,
                              num_classes=num_classes,
                              )

    def forward(self, features):

        #print ("features",features.shape)  
        #print (" before batch_norm", features[:10])
        if self.batchnorm: features = self.batchnorm(features)
        #print (" after batch_norm", features[:10])

        #print ("constituents_features",constituents_features.shape)  
        #print (" before conv", constituents_features[:10])
        #print (" after conv", features.shape, features[:10])
        #print ("points",points.shape, points[:10])
        return self.pn(features)
