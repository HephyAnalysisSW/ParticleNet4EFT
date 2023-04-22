import torch
import numpy as np
from math import pi
import array
import sys, os
sys.path.insert(0, '../..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers

import ROOT
dir_path = os.path.dirname(os.path.realpath(__file__))
import MicroMC 
model = MicroMC.make_model()

pt, angles = MicroMC.sample( model, 1 )
angles_ = angles[0]
Nsteps = 50
delay  = 50/10

plot_directory_ = os.path.join( user.plot_directory, 'EIRCGNN', "scatter")
os.makedirs( plot_directory_, exist_ok=True)

ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

p=ROOT.TColor.GetPalette()

for i in range(0, Nsteps):
    
    c1 = ROOT.TCanvas()
    angles_real = torch.view_as_real(angles_*np.exp(2*pi*1j*(i/Nsteps)))
    mask        = angles_real.abs().sum(dim=-1)!=0
    angles_real = angles_real[mask]

    x_max = torch.max(torch.abs(angles_)).item() 
    pt_log = np.log(array.array('d', pt[0][mask]))
    radii  = x_max/50.*(0.5 + (pt_log - np.min(pt_log)) / np.max(pt_log- np.min(pt_log)))
    colors = (254*(pt_log - np.min(pt_log)) / np.max(pt_log- np.min(pt_log))).astype(int) 

    stuff = []
    h = ROOT.TH2F('x','x',1,-x_max,x_max,1,-x_max,x_max)
    h.Draw("COLZ")
    for x, y, z, c, in zip(  array.array('d',angles_real[:,0]), array.array('d', angles_real[:,1]), radii, colors):
        stuff.append( ROOT.TEllipse(x, y, z) )
        stuff[-1].SetFillColorAlpha( p[c.item()], .55)
        stuff[-1].SetLineColorAlpha( p[c.item()], .55)
        stuff[-1].Draw()
    c1.Print(os.path.join( plot_directory_, 'test_%s.png'%(str(i).zfill(3))) )

    for o in stuff:
        del o 

helpers.copyIndexPHP( plot_directory_ )
syncer.makeRemoteGif(plot_directory_, pattern="test_*.png", name="test", delay=delay)
