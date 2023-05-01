import ctypes
import ROOT

class EFTModel:
    def __init__( self, thetaR, thetaG, events_per_parampoint=10):
        self.thetaR = thetaR
        self.thetaG = thetaG

        self.events_per_parampoint = events_per_parampoint

        # model: (exp(-R)+thetaR*exp(-R/alpha))^2 * (cos(y)*(1-thetaG) + sin(y)*thetaG)^2

        alpha       = 2.
        self.func               = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))**2*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaR        = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))*exp(-x/({alpha})) *(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaR_thetaR = ROOT.TF2("func", "exp(-2*x/({alpha})) *(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaG        = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))*(sin(y)-cos(y))".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaG_thetaG = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))*(sin(y)-cos(y))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaR_thetaG = ROOT.TF2("func", "                    exp(-x/({alpha})) *(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))*(sin(y)-cos(y))".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
    
    def getEvents( self, Nevents ):
   
        n1 = ctypes.c_double(0.) 
        n2 = ctypes.c_double(0.) 
        RPhi = []
        for _ in range( 1+Nevents//self.events_per_parampoint ):
            self.func.GetRandom2(n1, n2) 
            RPhi.append( [n1.value, n2.value] )

        weights = {tuple():[], ('thetaR',):[], ('thetaG',):[], ('thetaR','thetaR'):[], ('thetaG', 'thetaG'):[], ('thetaG', 'thetaR'):[]}

        pts, angles = [], []
        for iRPhi, (R, phi) in enumerate(RPhi):
            n = self.events_per_parampoint if Nevents-self.events_per_parampoint*iRPhi>=self.events_per_parampoint else Nevents%self.events_per_parampoint
            if n==0: break
            pt, angle = sample( make_model( R=R, phi=phi ), n )
            pts.append(torch.Tensor(pt))
            angles.append(torch.Tensor(angle))

            weights[tuple()].extend( [self.func.Eval(R, phi)]*n )
            weights[('thetaR',)].extend( [self.func_thetaR.Eval(R, phi)]*n )
            weights[('thetaR','thetaR')].extend( [self.func_thetaR_thetaR.Eval(R, phi)]*n )
            weights[('thetaG',)].extend( [self.func_thetaG.Eval(R, phi)]*n )
            weights[('thetaG', 'thetaG')].extend( [self.func_thetaG_thetaG.Eval(R, phi)]*n )
            weights[('thetaG', 'thetaR')].extend( [self.func_thetaR_thetaG.Eval(R, phi)]*n )

        return torch.cat( tuple(pts), dim=0), torch.cat( tuple(angles), dim=0), {key:torch.Tensor(val) for key, val in weights.items()}

    def __call__( self ):
        return 

#def loss( out, truth ):
#    return truth[:,0]*( out[0,:] - truth[] 
