import warnings
warnings.filterwarnings("ignore")

import ROOT
import os
import math
import numpy as np
import torch
import pickle
import array

import sys
sys.path.insert(0, '../..')
import tools.user as user
import tools.syncer as syncer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plot_directory',    action='store', default='v1', help="Prefix for training?")
parser.add_argument('--nTest', action='store', default=1000, type=int, help="Number of epochs.")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################### Load Model ################################
from SMEFTNet import SMEFTNet

models = [
    #{'dir':'def_conv_10_10__10_10_LFP', 'legendText':'10/10,10/10', 'color':ROOT.kBlue},
    #{'dir':'def_conv_10_10__20_20_LFP', 'legendText':'10/10,20/20', 'color':ROOT.kGreen},
    #{'dir':'def_conv_10_10__30_30_LFP', 'legendText':'10/10,30/30', 'color':ROOT.kRed},
    #{'dir':'def_conv_10_10__40_40_LFP', 'legendText':'10/10,40/40', 'color':ROOT.kOrange},
    #{'dir':'def_conv_10_10__50_50_LFP', 'legendText':'10/10,50/50', 'color':ROOT.kMagenta},

    {'dir':'def_conv_10_10_LFP', 'legendText':'10/10', 'color':ROOT.kBlue},
    {'dir':'def_conv_20_20_LFP', 'legendText':'20/20', 'color':ROOT.kGreen},
    {'dir':'def_conv_30_30_LFP', 'legendText':'30/30', 'color':ROOT.kRed},
    {'dir':'def_conv_40_40_LFP', 'legendText':'40/40', 'color':ROOT.kOrange},
    {'dir':'def_conv_50_50_LFP', 'legendText':'50/50', 'color':ROOT.kMagenta},

    #{'dir':'def_ro_10_10_LFP', 'legendText':'RO 10/10', 'color':ROOT.kBlue},
    #{'dir':'def_ro_20_20_LFP', 'legendText':'RO 20/20', 'color':ROOT.kGreen},
    #{'dir':'def_ro_30_30_LFP', 'legendText':'RO 30/30', 'color':ROOT.kRed},
    #{'dir':'def_ro_40_40_LFP', 'legendText':'RO 40/40', 'color':ROOT.kOrange},
    #{'dir':'def_ro_50_50_LFP', 'legendText':'RO 50/50', 'color':ROOT.kMagenta},

    #{'dir':'noPhi_conv_10_10__10_10', 'legendText':'10/10,10/10', 'color':ROOT.kBlue},
    #{'dir':'noPhi_conv_10_10__20_20', 'legendText':'10/10,20/20', 'color':ROOT.kGreen},
    #{'dir':'noPhi_conv_10_10__30_30', 'legendText':'10/10,30/30', 'color':ROOT.kRed},
    #{'dir':'noPhi_conv_10_10__40_40', 'legendText':'10/10,40/40', 'color':ROOT.kOrange},
    #{'dir':'noPhi_conv_10_10__50_50', 'legendText':'10/10,50/50', 'color':ROOT.kMagenta},

    #{'dir':'noPhi_conv_10_10', 'legendText':'10/10', 'color':ROOT.kBlue},
    #{'dir':'noPhi_conv_20_20', 'legendText':'20/20', 'color':ROOT.kGreen},
    #{'dir':'noPhi_conv_30_30', 'legendText':'30/30', 'color':ROOT.kRed},
    #{'dir':'noPhi_conv_40_40', 'legendText':'40/40', 'color':ROOT.kOrange},
    #{'dir':'noPhi_conv_50_50', 'legendText':'50/50', 'color':ROOT.kMagenta},

    #{'dir':'noPhi_ro_10_10', 'legendText':'RO 10/10', 'color':ROOT.kBlue},
    #{'dir':'noPhi_ro_20_20', 'legendText':'RO 20/20', 'color':ROOT.kGreen},
    #{'dir':'noPhi_ro_30_30', 'legendText':'RO 30/30', 'color':ROOT.kRed},
    #{'dir':'noPhi_ro_40_40', 'legendText':'RO 40/40', 'color':ROOT.kOrange},
    #{'dir':'noPhi_ro_50_50', 'legendText':'RO 50/50', 'color':ROOT.kMagenta},
]
#def_conv_10_10_LFP         
#def_conv_20_20_LFP         
#def_conv_30_30_LFP         
#def_conv_40_40_LFP
#def_conv_50_50_LFP
#def_ro_10_10_LFP
#def_ro_20_20_LFP
#def_ro_30_30_LFP
#def_ro_40_40_LFP
#def_ro_50_50_LFP
for model in models:
    model['model'] = SMEFTNet.load( os.path.join( user.model_directory, 'SMEFTNet', model['dir'] )).to(device)

torch.autograd.set_grad_enabled(False)

####################  Eval data ##########################
import MicroMC
from sklearn.model_selection import train_test_split
signal     = MicroMC.make_model( R=1, phi=0, var=0.3 )
background = MicroMC.make_model( R=1, phi=math.pi/2, var=0.3 )

pt_sig, angles_sig = MicroMC.sample(signal, args.nTest)
pt_bkg, angles_bkg = MicroMC.sample(background, args.nTest)
pt_sig      = torch.Tensor(pt_sig     ).to(device) 
angles_sig  = torch.Tensor(angles_sig ).to(device)
pt_bkg      = torch.Tensor(pt_bkg     ).to(device)
angles_bkg  = torch.Tensor(angles_bkg ).to(device)

########################## directories ###########################
plot_directory = os.path.join( user.plot_directory, 'SMEFTNet', 'roc', args.plot_directory )
os.makedirs( plot_directory, exist_ok=True)
print ("Using plot directory", plot_directory)
c1 = ROOT.TCanvas()
l  = ROOT.TLegend()
for i_model, model in enumerate(models): 
    out_sig  = model['model'](pt=pt_sig, angles=angles_sig)
    out_bkg  = model['model'](pt=pt_bkg, angles=angles_bkg)
    
    out_data = np.concatenate( (np.column_stack( (out_sig[:,0].numpy(), np.ones_like(out_sig[:,0]), np.zeros_like(out_sig[:,0]))), np.column_stack( (out_bkg[:,0].numpy(), np.zeros_like(out_bkg[:,0]), np.ones_like(out_bkg[:,0])))) )
    out_data = out_data[np.argsort(-out_data[:,0])]
    out_data = np.cumsum(out_data[:,1:],axis=0)
    out_data = np.array([((n_s/len(out_sig), n_b/len(out_bkg))) for n_s, n_b in out_data])

    model['roc'] = ROOT.TGraph(len(out_data), array.array('d', out_data[:,0]), array.array('d', out_data[:,1])) 
    if i_model==0:
        model['roc'].Draw('AL')
    else:
        model['roc'].Draw('L')
    model['roc'].SetLineColor( model['color'] )
    l.AddEntry( model['roc'], model['legendText'] )

l.Draw()
c1.Print(os.path.join( plot_directory, "roc.png"))
syncer.sync()
