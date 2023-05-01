import ctypes
import ROOT
from MicroMC import sample, make_model
import torch
import random
import math
class JetModel:
    def __init__( self, thetaR=0, thetaG=0, events_per_parampoint=10):
        #self.thetaR = thetaR
        #self.thetaG = thetaG

        self.events_per_parampoint = events_per_parampoint

        # model: (exp(-R)+thetaR*exp(-R/alpha))^2 * (cos(y)*(1-thetaG) + sin(y)*thetaG)^2

        #alpha       = 2.
        #self.func   = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))**2*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
    
    def getEvents( self, Nevents ):
   
        #n1 = ctypes.c_double(0.) 
        #n2 = ctypes.c_double(0.) 
        RPhi = []
        for _ in range( 1+Nevents//self.events_per_parampoint ):
            #self.func.GetRandom2(n1, n2) 
            #RPhi.append( [n1.value, n2.value] )
            RPhi.append( [1.5*random.random(), 2*math.pi*random.random()] )

        truth = {'R':[], 'gamma':[]}

        pts, angles = [], []
        for iRPhi, (R, phi) in enumerate(RPhi):
            n = self.events_per_parampoint if Nevents-self.events_per_parampoint*iRPhi>=self.events_per_parampoint else Nevents%self.events_per_parampoint
            if n==0: break
            pt, angle = sample( make_model( R=R, phi=phi ), n )
            pts.append(torch.Tensor(pt))
            angles.append(torch.Tensor(angle))

            truth['R'].append( R )
            truth['gamma'].append( phi )

        return torch.cat( tuple(pts), dim=0), torch.cat( tuple(angles), dim=0), torch.column_stack( tuple( torch.Tensor(truth[key]) for key in ['R', 'gamma']) )

    def __call__( self ):
        return 

def loss( out, truth ):
    return ( (out[:,0] - truth[:,0])**2 + (out[:,1]-truth[:,1])**2 ).sum() 
