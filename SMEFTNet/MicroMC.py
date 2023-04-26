import torch
import torch.distributions as D
import numpy as np
from math import pi, sin, cos

def make_model( R = 1, phi = 0, var = 0.3, two_prong = True):
    if two_prong:
        mix = D.Categorical(torch.ones(2,))
        comp = D.Independent(D.Normal(
                    torch.Tensor(((R*cos(phi), R*sin(phi)), (-R*cos(phi), -R*sin(phi)))),
                    torch.Tensor(((var,var),(var,var)))), 1)
        return D.MixtureSameFamily(mix, comp)
    else:
        comp = D.Normal(
                    torch.Tensor((R*cos(phi), R*sin(phi))),
                    torch.Tensor((var,var)) )
        return comp 

mean_Nparticles = 50
Nparticles      = D.Poisson(mean_Nparticles)
Nparticle_pad   = 80

def sample( model, Nevents ):
    pt_jet = 100
    smearing=0.1
    angles = torch.Tensor([torch.nn.functional.pad( model.sample((Nparticles.sample().int(),)), (0,0,0,Nparticle_pad))[:Nparticle_pad].numpy() for _ in range( Nevents )])
    mask=angles.abs().sum(dim=-1)!=0
    pt=torch.exp(model.log_prob(angles))
    pt = pt_jet*torch.distributions.log_normal.LogNormal(0,.2).sample((Nevents,Nparticle_pad))*(pt*mask)/(pt*mask).sum(dim=-1).reshape(-1,1)
    return pt.numpy(), angles.numpy()
