import warnings

warnings.filterwarnings("ignore")
import os
import math
import numpy as np
import torch
import pickle
import glob
#import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, global_max_pool
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

import sys
sys.path.insert(0, '../..')
import tools.user as user

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
parser.add_argument('--prefix',    action='store', default='v1', help="Prefix for training?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.002, help="Learning rate")
parser.add_argument('--small', action='store', default=None, type=int, help="Lear on a small subset?")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
parser.add_argument('--nTraining', action='store', default=100, type=int, help="Number of epochs.")
args = parser.parse_args()

class EdgeConv(MessagePassing):
    def __init__(self, mlp):
        super().__init__(aggr="sum") 
        self.mlp = mlp
 
    def forward(self, pt, features, angles, edge_index):

        #print( "pt", pt.shape) 
        #print( "features", features.shape) 
        #print( "angles", angles.shape) 
        return self.propagate(edge_index, x=torch.cat( (pt.view(-1,1), features, angles), dim=1), pt=pt)

    def message(self, x_i, x_j):
        # compute sin and cos(gamma_i-gamma_j) -> Here both rho and gamma drop out, only delta_gamma survives
        angles_ij  = torch.view_as_real(torch.view_as_complex(x_i[:,-2:].contiguous())/torch.view_as_complex(x_j[:,-2:].contiguous()))

        return torch.cat(  
                    (x_i[:,:1], #return pt of node 'i' .. (1 column) 
                         self.mlp(torch.cat(  # .. and mlp output as fkt of 
                                [x_i[:,1:-2], # features of node i  (1x nf(l))
                                 x_j[:,1:-2], # features of node j  (1x nf(l))
                                 x_j[:,1:-2]-x_i[:,1:-2], # differences (1x nf(l))
                                 angles_ij], dim=1)), #sin / cos(gamma_i - gamma_j) (2) -> 3 nf(l) + 2
                     x_i[:,-2:], # ... and finally the angles_norm of xi  (2 columns)
                     ), dim=1 ) 

    def aggregate( self, inputs, index):#,  **kwargs):
        ##remember: propagate calls message, aggregate, and update
        # inputs : ( pt, MLP[nf], angles[2]) 
        # where MLP[-1] is the phi_gamma
        # and MLP[:-1] are the features 
        # -> 1 + nf(l+1) + 1 + 2 = nf(l+1) + 4
        
        # we accumulate the pt according to the index and normalize
        pt          = inputs[:,0]
        wj = pt/( torch.zeros_like(index.unique(),dtype=torch.float).index_add_(0, index, pt)[index])

        # first, weight ALL inputs 
        result = torch.zeros((len(index.unique()),inputs.shape[1]),dtype=torch.float).to(device).index_add_(0, index, wj.view(-1,1)*inputs)
        # second, we take phi_gamma=MLP[-1]=inputs[-3] and equivariantly rotate the angles in what is now results[-2]. 
        # phi_gamma is not returned -> 1 + nf(l+1) + 2 -> nf(l+1) + 3
        result = torch.cat( (
            result[:,:-3], 
            torch.view_as_real( torch.exp( 2*torch.pi*1j*result[:,-3])*torch.view_as_complex(result[:,-2:].contiguous()) ),
            ), dim=1 )
            
        return result

from torch_geometric.nn.pool import radius

class EIRCGNN(EdgeConv):
    def __init__(self, mlp, dRN=0.4):
        super().__init__(mlp=mlp)
        self.dRN = dRN

    def forward(self, pt, features, angles, batch):
        #print ("pt",pt.shape, pt)
        #print ("angles", angles.shape, angles)
        #print ("features", features.shape, features)

        max_num_neighbors = max(batch.unique(return_counts=True)[1]).item()
        edge_index = radius(angles, angles, r=self.dRN, batch_x=batch, batch_y=batch, max_num_neighbors=max_num_neighbors)
        return super().forward(pt=pt, features=features, angles=angles, edge_index=edge_index)

dRN             = .4
conv_params     = ( (0.0, [5, 5, 4]),
                    (0.0, [5, 5, 3]))

readout_params = (0.0, [32,32])

norm_kwargs={'track_running_stats':False}
num_classes = 1

class Net(torch.nn.Module):
    def __init__(self, num_classes=num_classes, conv_params=conv_params, dRN=0.4, readout_params=readout_params):
        super().__init__()

        self.EC = torch.nn.ModuleList()

        for l, (dropout, hidden_layers) in enumerate(conv_params):
            hidden_layers_ = hidden_layers
            hidden_layers_[-1]+=1 # separate output for gamma-coordinate 
            if l==0:
                self.EC.append( EIRCGNN(MLP([3 + 2 ]+hidden_layers_, dropout=dropout, act="LeakyRelu"), dRN=dRN ) )
            else:
                self.EC.append( EIRCGNN(MLP([3*conv_params[l-1][1][-1]+2]+hidden_layers_,dropout=dropout, act="LeakyRelu"), dRN=dRN ) ) 

        EC_out_chn = hidden_layers_[-1]

        self.mlp = MLP( [EC_out_chn]+readout_params[1]+[num_classes], dropout=readout_params[0], act="LeakyRelu",norm_kwargs=norm_kwargs)

    def forward(self, pt, angles, batch):
    
        if len(self.EC)>0:

            for l, EC in enumerate(self.EC):
                x = EC( points if l==0 else x, features if idx==0 else x, batch)
            x = global_max_pool(x, batch)

        return self.mlp( x )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = Net().to(device)

################### TOY DATA #########################
import MicroMC
from sklearn.model_selection import train_test_split

signal     = MicroMC.make_model( R=1, phi=0, var=0.3 )
background = MicroMC.make_model( R=0, phi=0, var=0.3 )

pt_sig, angles_sig = MicroMC.sample(signal, args.nTraining)
pt_bkg, angles_bkg = MicroMC.sample(background, args.nTraining)

label_sig = np.ones( len(pt_sig) )
label_bkg = np.zeros( len(pt_bkg) )

pt_train, pt_test, angles_train, angles_test, labels_train, labels_test = train_test_split( 
        np.concatenate( (pt_sig, pt_bkg) ), 
        np.concatenate( (angles_sig, angles_bkg) ), 
        np.concatenate( (label_sig, label_bkg) )
    )

###################### TEST  ##########################
maxN = 2

phi=math.pi/4
R = torch.Tensor( [[math.cos(phi), math.sin(phi)],[-math.sin(phi), math.cos(phi)]] )

pt_train     = torch.Tensor(pt_train[:maxN]).to(device)
angles_train = torch.Tensor(angles_train[:maxN]).to(device)

mask = (angles_train.abs().sum(dim=-1) != 0)
labels_train = torch.Tensor(labels_train).to(device)

batch    = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]
pt_train = pt_train[mask]
angles_train = angles_train[mask]

#radius(angles_train, angles_train, r=0.4, batch_x=batch, batch_y=batch, max_num_neighbors=-1)

# We call the first EC layer with the pts (for aggregation) the only feature 'dR', and the full angular information for radius-aggregation
rho = torch.view_as_complex( angles_train ).abs()
result = model.EC[0].forward(pt=pt_train, features=rho.view(-1,1), angles=angles_train, batch=batch)

print ("Result of EC.forward:", result.shape)
print ("Result of EC.forward: result[0]", result[0])

assert False, ""



#optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
########################### directories ###########################
#
#model_directory = os.path.dirname( os.path.join( user.model_directory, 'DGCNN', args.prefix ))
#os.makedirs( model_directory , exist_ok=True)
#
############################ Scalers ##############################
#
#from sklearn.preprocessing import StandardScaler
#global_features_scaler = StandardScaler() if global_dims>0 else None
#y_scaler        = StandardScaler()
#features_scaler = StandardScaler()
#points_scaler   = StandardScaler()
#scaler_file = os.path.join( model_directory, 'scalers.pkl' )
#if not os.path.exists( scaler_file) or args.overwrite: 
#    for i_d, (X, y , observers, batch) in enumerate(training_data):
#        print(f'Fit scalers. At batch {i_d:03d}')
#        points   = torch.Tensor(X['points']).transpose(1,2)
#        features = torch.Tensor(X['features']).transpose(1,2)
#        mask     = (points.abs().sum(dim=-1) != 0)
#        features = features[mask]
#        points   = points[mask]
#        y = torch.Tensor(y['lin_coeff'])
#
#        points_scaler.partial_fit(points)
#        features_scaler.partial_fit(features)
#        y_scaler.partial_fit(y[:,1].reshape(-1,1))
#
#        if global_dims>0:
#            global_features = torch.Tensor(X['global_features'][:,:,0]) if global_dims>0 else None
#            global_features_scaler.partial_fit(global_features)
#    pickle.dump( (points_scaler, features_scaler, y_scaler, global_features_scaler), open(scaler_file, 'wb'))
#    print(f'Written scalers to {scaler_file:s}')
#else:
#    points_scaler, features_scaler, y_scaler, global_features_scaler = pickle.load(open(scaler_file, 'rb'))
#    print(f'Loaded scalers from {scaler_file:s}')
#
#
################# Loading previous state ###########################
#epoch_min = 0
#if not args.overwrite:
#    files = glob.glob( os.path.join( user.model_directory, 'DGCNN', args.prefix + '_epoch-*_state.pt') )
#    if len(files)>0:
#        load_file_name = max( files, key = lambda f: int(f.split('-')[-1].split('_')[0]))
#        load_epoch = int(load_file_name.split('-')[-1].split('_')[0])
#    else:
#        load_epoch = None
#    if load_epoch is not None:
#        print('Resume training from %s' % load_file_name)
#        model_state = torch.load(load_file_name, map_location=device)
#        model.load_state_dict(model_state)
#        opt_state_file = load_file_name.replace('_state.pt', '_optimizer.pt') 
#        if os.path.exists(opt_state_file):
#            opt_state = torch.load(opt_state_file, map_location=device)
#            optimizer.load_state_dict(opt_state)
#        else:
#            print('Optimizer state file %s NOT found!' % opt_state_file)
#        epoch_min=load_epoch+1
#
#####################  Training loop ##########################
#model.train()
#for epoch in range(epoch_min, args.epochs):
#
#    total_loss = 0
#    n_samples_total = 0
#
#    for i_d, (X, y , observers, _) in enumerate(training_data):
#
#        optimizer.zero_grad()
#
#        points   = torch.Tensor(X['points']).transpose(1,2).to(device)
#        features = torch.Tensor(X['features']).transpose(1,2).to(device)
#        mask = (points.abs().sum(dim=-1) != 0)
#        global_features = torch.Tensor(X['global_features'][:,:,0]).to(device) if global_dims>0 else None
#        y = torch.Tensor(y['lin_coeff']).to(device)
#        y_target = y[:,1]
#        y_weight = y[:,0]
#
#        features = features[mask]
#        points   = points[mask]
#        batch    = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]
#
#        features = (features-features_scaler.mean_.astype("float32"))/features_scaler.scale_.astype("float32")
#        points = (points-points_scaler.mean_.astype("float32"))/points_scaler.scale_.astype("float32")
#        if global_dims>0:
#            global_features = (global_features-global_features_scaler.mean_.astype("float32"))/global_features_scaler.scale_.astype("float32")
#
#        out = model(points, features, batch, global_features)
# 
#        #loss = y[:,0]*(out[:,0] - y[:,1])**2
#        #if i_d == 0:
#        loss  = (y_weight*criterion(out[:,0], y_target)).sum()
#        #else:
#        #    loss += (y_weight*criterion(out[:,0], y_target)).sum()
#        #loss_ = loss.item()
#        #total_loss+=loss.item()
#        n_samples = points.shape[0]
#        n_samples_total += n_samples
#        print(f'Epoch {epoch:03d}/{i_d:03d} with N={n_samples:03d}, Loss: {loss:.4f}')
#
#        loss.backward()
#        optimizer.step()
#
#    if args.prefix:
#        torch.save( model.state_dict(), os.path.join( user.model_directory, 'DGCNN', args.prefix + '_epoch-%d_state.pt' % epoch))
#        torch.save( optimizer.state_dict(), os.path.join( user.model_directory, 'DGCNN', args.prefix + '_epoch-%d_optimizer.pt' % epoch))
#        
#    print(f'Epoch {epoch:03d}, Ntot={n_samples_total:03d}')
#
#    model.eval()
#    with torch.no_grad():
#        observer_names = dataset.data_config.observers
#        obs_features   = {o:np.array([],dtype='float32') for o in observer_names}
#        truth          = {'lin':np.array([],dtype='float32')}
#        pred           = {'lin':np.array([],dtype='float32')}
#        w0             = np.array([],dtype='float32')
#
#        for X, y , observers, batch in training_data:
#            points   = torch.Tensor(X['points']).transpose(1,2).to(device)
#            features = torch.Tensor(X['features']).transpose(1,2).to(device)
#            mask = (points.abs().sum(dim=-1) != 0)
#            batch    = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]
#            global_features = torch.Tensor(X['global_features'][:,:,0]).to(device) if global_dims>0 else None
#
#
#            features = (features-features_scaler.mean_.astype("float32"))/features_scaler.scale_.astype("float32")
#            points = (points-points_scaler.mean_.astype("float32"))/points_scaler.scale_.astype("float32")
#            y_target = (y_target-y_scaler.mean_.astype("float32"))/y_scaler.scale_.astype("float32")
#            if global_dims>0:
#                global_features = (global_features-global_features_scaler.mean_.astype("float32"))/global_features_scaler.scale_.astype("float32")
#
#            out = model(points[mask], features[mask], batch, global_features)
#
#            # collect data for plotting 
#            for o in observer_names:
#                obs_features[o] = np.concatenate( (obs_features[o], observers[o]) )
#            pred['lin']  = np.concatenate( (pred['lin'], out[:,0].cpu() ))
#            truth['lin'] = np.concatenate( (truth['lin'], y['lin_coeff'][:,1] ))
#            w0           = np.concatenate( (w0, y['lin_coeff'][:,0] ))
#        plot.plot( epoch=epoch, features=[(key, obs_features[key]) for key in observer_names], w0=w0, prediction=pred, truth=truth, 
#                   plot_options=dataset.plot_options,
#                   plot_directory='DGCNN_%s'%args.prefix + ('_small_%i'%args.small if args.small is not None else '') ,
#                   do2D = False, 
#                )
