import warnings

warnings.filterwarnings("ignore")
import os
import math
import numpy as np
import torch
import pickle
import glob
import copy

# reproducibility
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

from torch_geometric.nn import MessagePassing, MLP

import sys
sys.path.insert(0, '../..')
import tools.user as user

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
parser.add_argument('--prefix',    action='store', default='v1', help="Prefix for training?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.001, help="Learning rate")
parser.add_argument('--small', action='store', default=None, type=int, help="Lear on a small subset?")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
parser.add_argument('--nTraining', action='store', default=1000, type=int, help="Number of epochs.")
args = parser.parse_args()

class EdgeConv(MessagePassing):
    def __init__(self, mlp):
        super().__init__(aggr="sum") 
        self.mlp = mlp

        # log messages
        self.message_logging = False
        self.message_dict    = {}
 
    def forward(self, x, edge_index):
        with torch.no_grad():
            if self.message_logging:
                self.message_dict["edge_index"] = edge_index 
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):

        #print( x_i[:,-2:] )
        #print( x_i[:,-2:].shape )
        #print( x_i[:,-2:].stride() )
        #print( torch.view_as_complex(x_i[:,-2:]) )

        #print( "an i", x_i[:,-2:] )
        #print( "an j", x_j[:,-2:] )
        # compute sin and cos(gamma_i-gamma_j) -> Here both rho and gamma drop out, only delta_gamma survives

        #angles_ij  = torch.view_as_real(torch.view_as_complex(x_i[:,-2:].contiguous())/torch.view_as_complex(x_j[:,-2:].contiguous()))
        norm = torch.sqrt((x_i[:,-2]**2+ x_i[:,-1]**2 )*(x_j[:,-2]**2+ x_j[:,-1]**2))
        cos_ij, sin_ij = ( (x_i[:,-2]*x_j[:,-2] + x_i[:,-1]*x_j[:,-1])/norm, (x_i[:,-1]*x_j[:,-2]-x_i[:,-2]*x_j[:,-1])/norm )
        mlp = self.mlp(torch.cat(  # .. and mlp output as fkt of 
                [x_i[:,1:-2], # features of node i  (1x nf(l))
                 x_j[:,1:-2], # features of node j  (1x nf(l))
                 x_j[:,1:-2]-x_i[:,1:-2], # differences (1x nf(l))
                 cos_ij.view(-1,1), # and two angular coordinates, in / cos(gamma_i - gamma_j) 
                 sin_ij.view(-1,1), # -> 3 nf(l) + 2
                ], dim=1)) 
        if torch.any(torch.isnan(mlp)):
            print ("Warning! Found nan in message passing MLP output. Set to zero (likely zero variance).")
            mlp = torch.nan_to_num(mlp) 
        return torch.cat(  
            ( x_i[:,:1], #return pt of node 'i' .. (1 column)
              mlp, 
              x_i[:,-2:], # ... and finally the angles of xi  (2 columns)
             ), dim=1 ) 

    def aggregate( self, inputs, index):#,  **kwargs):
        ##remember: propagate calls message, aggregate, and update
        # inputs : ( pt, MLP[nf], angles[2]) 
        # where MLP[-1] is the phi_gamma
        # and MLP[:-1] are the features 
        # -> 1 + nf(l+1) + 1 + 2 = nf(l+1) + 4
        
        # we accumulate the pt according to the index and normalize (IRC safe pooling of messages)

        pt = inputs[:,0]
        #print ("inputs", inputs.shape)
        #print ("index", index.shape)
        #print ("pt", pt.shape)
        wj = pt/( torch.zeros_like(index.unique(),dtype=torch.float).index_add_(0, index, pt)[index])
        if torch.any( torch.isnan(wj)):
            print ("Warning! Found nan in pt weighted message passing (aggregation). There is a particle with only pt=0 particles in its neighbourhood. Replace with zero.")
            wj = torch.nan_to_num(wj)

        # first, weight ALL inputs
        result = torch.zeros((len(index.unique()),inputs.shape[1]),dtype=torch.float).to(device).index_add_(0, index, wj.view(-1,1)*inputs)
        # second, we take phi_gamma=MLP[-1]=inputs[-3] and equivariantly rotate the angles in what is now results[-2]. 
        # phi_gamma is not returned -> 1 + nf(l+1) + 2 -> nf(l+1) + 3
        #print ("EC out", result.shape)            
        result = torch.cat( (
            result[:,:-3], 
            torch.view_as_real( torch.exp( 2*torch.pi*1j*result[:,-3])*torch.view_as_complex(result[:,-2:].contiguous()) ),
            ), dim=1 )
        #if True:
        with torch.no_grad():
            if self.message_logging:
                self.message_dict["message"] = torch.sqrt( torch.square(inputs[:, 1:-3]).sum(dim=-1)).numpy() 
            
        #print ("result EC", result.shape, result)
        #print ("index", index)
        #print ("index", index.unique())
        return result

from torch_geometric.nn.pool import radius

class EIRCGNN(EdgeConv):
    def __init__(self, mlp, dRN=0.4):
        super().__init__(mlp=mlp)
        self.dRN = dRN

    def forward(self, x, batch):

        # ( pt[1], features, angles[2] )
        #print ("pt",pt.shape, pt)
        #print ("features", features.shape, features)
        #print ("angles", angles.shape, angles)

        max_num_neighbors = max(batch.unique(return_counts=True)[1]).item()
        edge_index = radius(x[:,-2:], x[:,-2:], r=self.dRN, batch_x=batch, batch_y=batch, max_num_neighbors=max_num_neighbors)
        return super().forward(x, edge_index=edge_index)

dRN             = .4
conv_params     = ( 
                    (0.0, [20, 20]),
#                    (0.0, [5, 5, 2])
                    )
readout_params = (0.0, [32,32])

norm_kwargs={}#'track_running_stats':False}
num_classes = 1

class Net(torch.nn.Module):
    def __init__(self, num_classes=num_classes, conv_params=conv_params, dRN=0.4, readout_params=readout_params):
        super().__init__()

        self.EC = torch.nn.ModuleList()

        for l, (dropout, hidden_layers) in enumerate(conv_params):
            hidden_layers_ = copy.deepcopy(hidden_layers)
            hidden_layers_[-1]+=1 # separate output for gamma-coordinate 
            if l==0:
                self.EC.append( EIRCGNN(MLP([3 + 2 ]+hidden_layers_, dropout=dropout, act="LeakyRelu"), dRN=dRN ) )
            else:
                self.EC.append( EIRCGNN(MLP([3*conv_params[l-1][1][-1]+2]+hidden_layers_,dropout=dropout, act="LeakyRelu"), dRN=dRN ) ) 

        # output features + sin/cos gamma
        EC_out_chn = hidden_layers[-1]

        self.mlp = MLP( [EC_out_chn]+readout_params[1]+[num_classes], dropout=readout_params[0], act="LeakyRelu",norm_kwargs=norm_kwargs)
        self.out = torch.nn.Sigmoid()

    def forward(self, pt, angles, message_logging=False):

        # for IRC tests we actually low zero pt. Zero abs angles define the mask
        mask = (angles.abs().sum(dim=-1) != 0)
        batch= (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]

        # we feed pt in col. 0, rho (as feature) in col. 1, and then the angles in col. 2,3
        x = torch.cat( (pt[mask].view(-1,1), torch.view_as_complex( angles[mask] ).abs().view(-1,1), angles[mask]), dim=1) 
        for l, EC in enumerate(self.EC):
            EC.message_logging = message_logging
            x = EC(x, batch)

        # IRC safe pooling
        pt = x[:,0] 
        #return pt, batch 
        wj = pt/( torch.zeros_like(batch.unique(),dtype=torch.float).index_add_(0, batch, pt))[batch]
        if torch.any( torch.isnan(wj)):
            print ("Warning! Found nan in pt weighted readout. Are there no particles with pt>0?. Replace with zero.")
            wj = torch.nan_to_num(wj)
        #print ("wj",wj.shape)
        #print ("x",x.shape)
        # disregard first column (pt, keep the last two ones: sin/cos gamma)
        x = torch.zeros((len(batch.unique()),x[:,1:].shape[1]),dtype=torch.float).to(device).index_add_(0, batch, wj.view(-1,1)*x[:,1:])
        
        return torch.cat( (self.out(self.mlp( x[:, :-2] )), x[:, -2:]), dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = Net().to(device)

################ Micro MC Toy Data #####################
import MicroMC
from sklearn.model_selection import train_test_split

###################### TESTS  ##########################
#signal     = MicroMC.make_model( R=1, phi=0, var=0.3 )
#background = MicroMC.make_model( R=0, phi=0, var=0.3 )
#
#pt_sig, angles_sig = MicroMC.sample(signal, 100)
#pt_bkg, angles_bkg = MicroMC.sample(background, 100)
#
#label_sig = np.ones(  len(pt_sig) )
#label_bkg = np.zeros( len(pt_bkg) )
#
#pt_train, pt_test, angles_train, angles_test, labels_train, labels_test = train_test_split( 
#        np.concatenate( (pt_sig, pt_bkg) ), 
#        np.concatenate( (angles_sig, angles_bkg) ), 
#        np.concatenate( (label_sig, label_bkg) )
#    )
#maxN = 1
#pt_train     = torch.Tensor(pt_train[:maxN]).to(device)
#angles_train = torch.Tensor(angles_train[:maxN]).to(device)
#labels_train = torch.Tensor(labels_train[:maxN]).to(device)
#model.eval()
#result = model( pt=pt_train, angles=angles_train)
#
#with torch.no_grad():
#    result = model( pt=pt_train, angles=angles_train)
##################### EQUIVARIANCE #######################
    #for i in range(101):
    #    phi = 2*math.pi*i/100
    #    R   = torch.Tensor( [[math.cos(phi), math.sin(phi)],[-math.sin(phi), math.cos(phi)]] )
    #    angles_train_ = torch.matmul( angles_train, R)

    #    #rho = torch.view_as_complex( angles_train ).abs()
    #    result = model( pt=pt_train, angles=angles_train_)
    #    classifier, angles = result[:,:-2], result[:,-2:]
    #    print ("classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
#################### IR safety #######################
    # add a bunch of soft particles
    #pt_train     = torch.Tensor([[1.]]).to(device)
    #angles_train = torch.Tensor([[[.5, .5]]]).to(device)
    #result = model( pt=pt_train, angles=angles_train)
    #classifier, angles = result[:,:-2], result[:,-2:]
    #print ("orig classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )

    #result = model( pt=pt_train, angles=angles_train)
    #classifier, angles = result[:,:-2], result[:,-2:]
    #print ("orig classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item() )

    #for i in range(-3,9):
    #    pt_soft = 10**(-i)
    #    pt_train     = torch.Tensor([[1., pt_soft, pt_soft, pt_soft]]).to(device)
    #    angles_train = torch.Tensor([[[.5, .5], [1., 0.], [-.3,.4], [5,-.6]]]).to(device)
    #    result = model( pt=pt_train, angles=angles_train)
    #    classifier, angles = result[:,:-2], result[:,-2:]
    #    print ("classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
#################### Collinear safety #######################
#
#    pt_train     = torch.Tensor([[1., 2.]]).to(device)
#    angles_train = torch.Tensor([[[.5, .5], [-.3,.3]]]).to(device)
#    result = model( pt=pt_train, angles=angles_train)
#    classifier, angles = result[:,:-2], result[:,-2:]
#    print ("orig classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
#
#    for i in range(0,11):
#        l = i/10. 
#        pt_train     = torch.Tensor([[1., 2*l, 2*(1-l)]]).to(device)
#        angles_train = torch.Tensor([[[.5, .5], [-.3, .3], [-.3, .3]]]).to(device)
#        result = model( pt=pt_train, angles=angles_train)
#        classifier, angles = result[:,:-2], result[:,-2:]
#        print ("i",i,"classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
#
#assert False, ""


optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1./20)

#def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
#        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
#    return _log_cosh(y_pred - y_true)
#
#class LogCoshLoss(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#
#    def forward(
#        self, y_pred: torch.Tensor, y_true: torch.Tensor
#    ) -> torch.Tensor:
#        return log_cosh_loss(y_pred, y_true)
#
#criterion = LogCoshLoss() 
#
########################## directories ###########################

model_directory = os.path.dirname( os.path.join( user.model_directory, 'EIRCGNN', args.prefix ))
os.makedirs( model_directory , exist_ok=True)

################ Loading previous state ###########################
epoch_min = 0
if not args.overwrite:
    files = glob.glob( os.path.join( user.model_directory, 'EIRCGNN', args.prefix + '_epoch-*_state.pt') )
    if len(files)>0:
        load_file_name = max( files, key = lambda f: int(f.split('-')[-1].split('_')[0]))
        load_epoch = int(load_file_name.split('-')[-1].split('_')[0])
    else:
        load_epoch = None
    if load_epoch is not None:
        print('Resume training from %s' % load_file_name)
        model_state = torch.load(load_file_name, map_location=device)
        model.load_state_dict(model_state)
        opt_state_file = load_file_name.replace('_state.pt', '_optimizer.pt') 
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            optimizer.load_state_dict(opt_state)
        else:
            print('Optimizer state file %s NOT found!' % opt_state_file)
        epoch_min=load_epoch+1

####################  Training loop ##########################
signal     = MicroMC.make_model( R=1, phi=0, var=0.3 )
background = MicroMC.make_model( R=0, phi=0, var=0.3 )

def getEvents( nTraining=args.nTraining ):

    pt_sig, angles_sig = MicroMC.sample(signal, nTraining)
    pt_bkg, angles_bkg = MicroMC.sample(background, nTraining)

    label_sig = torch.ones(  len(pt_sig) )
    label_bkg = torch.zeros( len(pt_bkg) )
    return train_test_split( 
        torch.Tensor(np.concatenate( (pt_sig, pt_bkg) )), 
        torch.Tensor(np.concatenate( (angles_sig, angles_bkg) )), 
        torch.Tensor(np.concatenate( (label_sig, label_bkg) ))
    )

criterion = torch.nn.BCELoss()
model.train()
pt_train, pt_test, angles_train, angles_test, labels_train, labels_test = getEvents(args.nTraining) 
for epoch in range(epoch_min, args.epochs):

    optimizer.zero_grad()

    out  = model(pt=pt_train, angles=angles_train)
    loss = criterion(out[:,0], labels_train )
    n_samples = len(pt_train) 
    print(f'Epoch {epoch:03d} with N={n_samples:03d}, Loss: {loss:.4f}')

    loss.backward()
    optimizer.step()

    if args.prefix:
        torch.save( model.state_dict(), os.path.join( user.model_directory, 'EIRCGNN', args.prefix + '_epoch-%d_state.pt' % epoch))
        torch.save( optimizer.state_dict(), os.path.join( user.model_directory, 'EIRCGNN', args.prefix + '_epoch-%d_optimizer.pt' % epoch))

model.eval()
model( pt_test[:2], angles_test[:2] )

#    with torch.no_grad():
#        out = model(pt=pt_test, angles=angles_test)
