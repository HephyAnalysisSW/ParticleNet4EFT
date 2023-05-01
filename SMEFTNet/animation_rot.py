import torch
import numpy as np
import math 
import array
import sys, os
sys.path.insert(0, '../..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers

import ROOT
dir_path = os.path.dirname(os.path.realpath(__file__))

# reproducibility
torch.manual_seed(1)
import numpy as np
np.random.seed(0)

from sklearn.model_selection import train_test_split
import MicroMC
signal     = MicroMC.make_model( R=1, phi=0, var=0.3 )
background = MicroMC.make_model( R=1, phi=math.pi/2, var=0.3 )

signal_pdf = MicroMC.make_TH2(R=1, phi=0, var=0.3)
bkg_pdf    = MicroMC.make_TH2(R=1, phi=math.pi/2, var=0.3)

nTest = 1
pt_sig, angles_sig = MicroMC.sample(signal, nTest)
pt_bkg, angles_bkg = MicroMC.sample(background, nTest)
pt_sig      = torch.Tensor(pt_sig     ) 
angles_sig  = torch.Tensor(angles_sig )
pt_bkg      = torch.Tensor(pt_bkg     )
angles_bkg  = torch.Tensor(angles_bkg )

angles_plot = angles_bkg[0]
pt_plot     = pt_bkg[0]

Nsteps = 100
delay  = 50/5

plot_directory_ = os.path.join( user.plot_directory, 'SMEFTNet', "scatter")
os.makedirs( plot_directory_, exist_ok=True)

ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Load model
from SMEFTNet import SMEFTNet
model_dir  = 'def_conv_10_10_LFP'
data_model = 'R1dGamma'
model      = SMEFTNet.load( os.path.join( user.model_directory, 'SMEFTNet', data_model, model_dir ))
torch.autograd.set_grad_enabled(False)

# number of features before the final MLP
n_h = eval(model.cfg_dict['conv_params'])[-1][1][-1] 
# Two more features in the output: cos/sin of the angle
h2  = ROOT.TH1F('h','h', n_h+3, 0, n_h+3)
for i_bin in range(1, 1+h2.GetNbinsX()):
    if i_bin<h2.GetNbinsX():
        h2.GetXaxis().SetBinLabel( i_bin, "h_{%i}"%i_bin )

h2.GetXaxis().SetBinLabel( h2.GetNbinsX()-2, "cos(\gamma_{\phi})")
h2.GetXaxis().SetBinLabel( h2.GetNbinsX()-1, "sin(\gamma_{\phi})")
h2.GetXaxis().SetBinLabel( h2.GetNbinsX(), "out")

for i_bin in range(-2+h2.GetNbinsX(), 1+h2.GetNbinsX()):
    h2.GetXaxis().ChangeLabel( i_bin, 270. )

p=ROOT.TColor.GetPalette()
width = 500
width = 500
c1 = ROOT.TCanvas("c", "c", 200, 10, 2*width, width)
c1.Divide(2,1)
gray_circles = []
for i in range(0, Nsteps):
    
    angles_real = torch.view_as_real(torch.view_as_complex(angles_plot)*np.exp(2*math.pi*1j*(i/Nsteps)))
    
    mask        = angles_real.abs().sum(dim=-1)!=0
    angles_real_ = angles_real[mask]

    x_max = 1.2*torch.max(torch.abs(angles_plot)).item() 
    pt_log = np.log(array.array('d', pt_plot[mask]))
    radii  = x_max/50.*(0.5 + (pt_log - np.min(pt_log)) / np.max(pt_log- np.min(pt_log)))
    colors = (254*(pt_log - np.min(pt_log)) / np.max(pt_log- np.min(pt_log))).astype(int) 

    c1.cd(1)
    h = ROOT.TH2F('x','x',1,-x_max,x_max,1,-x_max,x_max)
    h.Draw("COLZ")
    h.GetXaxis().SetTitle("#Delta y")
    h.GetYaxis().SetTitle("#Delta #phi")

    bkg_pdf.SetLineStyle(2)

    #signal_pdf.SetContour( 5 )
    signal_pdf.SetLineColor(ROOT.kGray)
    bkg_pdf   .SetLineColor(ROOT.kGray)
    signal_pdf.SetLineWidth(1)
    bkg_pdf   .SetLineWidth(1)

    signal_pdf.Draw("CONT3same")
    bkg_pdf.Draw("CONT3same")

    circles = []
    for x, y, z, c, in zip(  array.array('d',angles_real_[:,0]), array.array('d', angles_real_[:,1]), radii, colors):
        circles.append( ROOT.TEllipse(x, y, z) )
        circles[-1].SetFillColorAlpha( p[c.item()], .55)
        circles[-1].SetLineColorAlpha( p[c.item()], .55)

        if i==0:
            gray_circles.append( ROOT.TEllipse(x, y, z) )
            gray_circles[-1].SetFillColorAlpha( ROOT.kGray, .55)
            gray_circles[-1].SetLineColorAlpha( ROOT.kGray, .55)
            gray_circles[-1].Draw()

    for o in gray_circles + circles:
        o.Draw()

    legend = ROOT.TLegend(0.18, 0.83, 0.45, 0.93)
    #legend.SetNColumns(legendColumns)
    legend.SetFillStyle(0)
    legend.SetShadowColor(ROOT.kWhite)
    legend.SetBorderSize(0)

    signal_pdf.SetFillStyle(0)
    bkg_pdf.SetFillStyle(0)

    legend.AddEntry( signal_pdf, "Signal PDF" )
    legend.AddEntry( bkg_pdf, "Background PDF" )
    legend.Draw()

    c1.cd(2)

    out = model(pt=pt_plot.view(1,-1), angles=angles_real.view(1,-1,2))
    print (out)
    h2.SetBinContent( h2.GetNbinsX(), out[0][0].item() )
    h3 = h2.Clone()
    h4 = h2.Clone()
    
    #c1.Print(os.path.join( plot_directory_, 'test_%s.png'%(str(i).zfill(3))) )
    out_EIRCGNN = model.EIRCGNN_output(pt=pt_plot.view(1,-1),angles=angles_real.view(1,-1,2))
    phi_scale   = math.sqrt(out_EIRCGNN[0][-2]**2 + out_EIRCGNN[0][-1]**2)

    for i_bin in range( 1, h2.GetNbinsX()):
        h2.SetBinContent(i_bin, (1./phi_scale if i_bin in [h2.GetNbinsX()-2, h2.GetNbinsX()-1] else 1)*out_EIRCGNN[0][i_bin-1].item())
        if i_bin in  [h2.GetNbinsX()-2, h2.GetNbinsX()-1]:
            h4.SetBinContent(i_bin, 1./phi_scale*out_EIRCGNN[0][i_bin-1].item())

    h2.Draw("hist")

    h3.SetFillColor(ROOT.kCyan-8)
    h4.SetFillColor(ROOT.kCyan-9)
    h4.Draw("same")
    h3.Draw("same")

    h2.Draw("histsame")
    h2.GetYaxis().SetRangeUser(-1.8,1.8)

    l1 = ROOT.TLine(h2.GetXaxis().GetBinLowEdge(h2.GetNbinsX()), h2.GetMinimum(), h2.GetXaxis().GetBinLowEdge(h2.GetNbinsX()), h2.GetMaximum())
    l2 = ROOT.TLine(h2.GetXaxis().GetBinLowEdge(h2.GetNbinsX()-2), h2.GetMinimum(), h2.GetXaxis().GetBinLowEdge(h2.GetNbinsX()-2), h2.GetMaximum())

    l1.SetLineStyle(2)
    l2.SetLineStyle(2)
    l1.Draw()
    l2.Draw()

    c1.Print(os.path.join( plot_directory_, 'test_%s.png'%(str(i).zfill(3))) )

    h2.Reset()
    h3.Reset()
    h4.Reset()

helpers.copyIndexPHP( plot_directory_ )
syncer.sync()
syncer.makeRemoteGif(plot_directory_, pattern="test_*.png", name="test", delay=delay)
