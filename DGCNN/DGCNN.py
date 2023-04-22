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

import plot

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
parser.add_argument('--prefix',    action='store', default='ctGRe', help="Prefix for training?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.002, help="Learning rate")
parser.add_argument('--small', action='store', default=None, type=int, help="Lear on a small subset?")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
args = parser.parse_args()

import dataset

training_data= dataset.dataset if not args.small else dataset.Dataset(dataset.directory, n_split=1, max_n_files=1, max_events=args.small)

feature_dims = len(dataset.data_config.input_dicts['features']) if 'features' in dataset.data_config.input_dicts else 0
point_dims   = len(dataset.data_config.input_dicts['points']) if 'points' in dataset.data_config.input_dicts else 0
global_dims  = 0#len(dataset.data_config.inputs['global_features']['vars']) if 'global_features' in dataset.data_config.input_dicts else 0


class EdgeConv(MessagePassing):
    def __init__(self, mlp, aggr):
        super().__init__(aggr=aggr) #  "Max" aggregation.
        self.mlp = mlp
 
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        #return self.mlp(tmp)
        return self.mlp(tmp)

from torch_geometric.nn import knn_graph

class DynamicEdgeConv(EdgeConv):
    def __init__(self, mlp, k, aggr):
        super().__init__(mlp, aggr)
        self.k = k

    def forward(self, points, features, batch=None):
        edge_index = knn_graph(points, self.k, batch, loop=False, flow=self.flow)
        return super().forward(features, edge_index)

class FeatureConv(torch.nn.Module):
    def __init__(self, in_chn, out_chn, batch_norm=True, activation=True):
        super(FeatureConv, self).__init__()
        if batch_norm:
            self.feature_conv = torch.nn.Sequential(
                torch.nn.BatchNorm1d(in_chn),
                torch.nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                torch.nn.BatchNorm1d(out_chn),
                )
        else:
            self.feature_conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                )
        if activation:
            self.feature_conv.append( torch.nn.ReLU() )

    def forward(self, x):
        return self.feature_conv(x)

size     = 32
k        = 20

conv_params = [ 
    (k, [size, size, size]),
    (k, [2*size, 2*size])
]

fusion_output = None#8*size

norm_kwargs={'track_running_stats':False}

num_classes = 1
conv_dim = 16
global_params    = ( [4*size, 4*size], 0.0 ) 
combined_params  = ( [4*size, 4*size], 0.0 ) 

class Net(torch.nn.Module):
    def __init__(self, out_channels=1, aggr='mean', global_dims=0):
        super().__init__()

        dEC_feature_dims = feature_dims
        if conv_dim>0 and feature_dims>0 and len(conv_params)>0:
            self.feature_conv = FeatureConv(feature_dims, conv_dim, batch_norm=True, activation=True)
            dEC_feature_dims  = conv_dim
        else:
            self.feature_conv = None

        #n_point_dim = 2
        #self.dEC1 = DynamicEdgeConv(MLP([2 * n_point_dim, size, size, size]), k, aggr)
        #self.dEC2 = DynamicEdgeConv(MLP([2 * size, 2*size]), k, aggr)
        #self.lin1 = Linear(3*size, 8*size)
        #dEC_out_chn = 8*size
        
        self.dEC = torch.nn.ModuleList()
        dEC_out_chn_total = 0
        dEC_out_chn       = 0

        for idx, (k, conv_param ) in enumerate(conv_params):
            dEC_out_chn_total+=conv_param[-1]
            if idx==0:
                self.dEC.append( DynamicEdgeConv(MLP([2*feature_dims]+conv_param), k, aggr) )
            else:
                self.dEC.append( DynamicEdgeConv(MLP([2*conv_params[idx-1][1][-1]]+conv_param), k, aggr) )
                dEC_out_chn = conv_param[-1]

        self.fusion = fusion_output is not None and (fusion_output>0)
        if self.fusion:
            self.lin1 = Linear(dEC_out_chn_total, fusion_output)
            dEC_out_chn = fusion_output
        else:
            self.lin1 = None

        # MLP for global features
        self.global_dims  = global_dims
        global_out_chn    = self.global_dims
        if self.global_dims>0 and global_params is not None:
            self.global_mlp = MLP( [self.global_dims]+global_params[0], dropout=global_params[1],norm_kwargs=norm_kwargs)
            global_out_chn  = global_params[0][-1]    
        else:
            self.global_mlp = None
            global_out_chn  = self.global_dims

        # combined MLP
        self.mlp = MLP( [dEC_out_chn+global_out_chn]+combined_params[0]+[num_classes], dropout=combined_params[1], norm=None,norm_kwargs=norm_kwargs)

    def forward(self, points, features, batch, global_features):
    
        #if self.feature_conv:
        #    x = self.feature_conv(features) 
        #else:
        #    x = features 
        #x1 = self.dEC1(points, batch)
        #x2 = self.dEC2(x1, batch)
        #x = self.lin1(torch.cat([x1, x2], dim=1))

        if len(self.dEC)>0:

            outputs = []
            for idx, dEC in enumerate(self.dEC):
                x = dEC( points if idx==0 else x, features if idx==0 else x, batch)
                if self.fusion:
                    outputs.append( x )
            if self.fusion:
                x = self.lin1( torch.cat(outputs, dim=1))
            x = global_max_pool(x, batch)

        # global features
        if self.global_mlp:
            y = self.global_mlp( global_features)
        else:
            y = global_features

        if len(self.dEC)>0:
            if self.global_dims>0:
                return self.mlp(torch.cat( (x,y), dim=1 ))
            else:
                return self.mlp( x )
        else:
            if self.global_dims>0:
                return self.mlp( y )
            else:
                raise RuntimeError( "Neither EdgeConv nor global features provided." ) 

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = Net(out_channels=1, global_dims=global_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return _log_cosh(y_pred - y_true)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

criterion = LogCoshLoss() 

########################## directories ###########################

model_directory = os.path.dirname( os.path.join( user.model_directory, 'DGCNN', args.prefix ))
os.makedirs( model_directory , exist_ok=True)

########################### Scalers ##############################

from sklearn.preprocessing import StandardScaler
global_features_scaler = StandardScaler() if global_dims>0 else None
y_scaler        = StandardScaler()
features_scaler = StandardScaler()
points_scaler   = StandardScaler()
scaler_file = os.path.join( model_directory, 'scalers.pkl' )
if not os.path.exists( scaler_file) or args.overwrite: 
    for i_d, (X, y , observers, batch) in enumerate(training_data):
        print(f'Fit scalers. At batch {i_d:03d}')
        points   = torch.Tensor(X['points']).transpose(1,2)
        features = torch.Tensor(X['features']).transpose(1,2)
        mask     = (points.abs().sum(dim=-1) != 0)
        features = features[mask]
        points   = points[mask]
        y = torch.Tensor(y['lin_coeff'])

        points_scaler.partial_fit(points)
        features_scaler.partial_fit(features)
        y_scaler.partial_fit(y[:,1].reshape(-1,1))

        if global_dims>0:
            global_features = torch.Tensor(X['global_features'][:,:,0]) if global_dims>0 else None
            global_features_scaler.partial_fit(global_features)
    pickle.dump( (points_scaler, features_scaler, y_scaler, global_features_scaler), open(scaler_file, 'wb'))
    print(f'Written scalers to {scaler_file:s}')
else:
    points_scaler, features_scaler, y_scaler, global_features_scaler = pickle.load(open(scaler_file, 'rb'))
    print(f'Loaded scalers from {scaler_file:s}')


################ Loading previous state ###########################
epoch_min = 0
if not args.overwrite:
    files = glob.glob( os.path.join( user.model_directory, 'DGCNN', args.prefix + '_epoch-*_state.pt') )
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
model.train()
for epoch in range(epoch_min, args.epochs):

    total_loss = 0
    n_samples_total = 0

    for i_d, (X, y , observers, _) in enumerate(training_data):

        optimizer.zero_grad()

        points   = torch.Tensor(X['points']).transpose(1,2).to(device)
        features = torch.Tensor(X['features']).transpose(1,2).to(device)
        mask = (points.abs().sum(dim=-1) != 0)
        global_features = torch.Tensor(X['global_features'][:,:,0]).to(device) if global_dims>0 else None
        y = torch.Tensor(y['lin_coeff']).to(device)
        y_target = y[:,1]
        y_weight = y[:,0]

        features = features[mask]
        points   = points[mask]
        batch    = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]

        features = (features-features_scaler.mean_.astype("float32"))/features_scaler.scale_.astype("float32")
        points = (points-points_scaler.mean_.astype("float32"))/points_scaler.scale_.astype("float32")
        if global_dims>0:
            global_features = (global_features-global_features_scaler.mean_.astype("float32"))/global_features_scaler.scale_.astype("float32")

        out = model(points, features, batch, global_features)
 
        #loss = y[:,0]*(out[:,0] - y[:,1])**2
        #if i_d == 0:
        loss  = (y_weight*criterion(out[:,0], y_target)).sum()
        #else:
        #    loss += (y_weight*criterion(out[:,0], y_target)).sum()
        #loss_ = loss.item()
        #total_loss+=loss.item()
        n_samples = points.shape[0]
        n_samples_total += n_samples
        print(f'Epoch {epoch:03d}/{i_d:03d} with N={n_samples:03d}, Loss: {loss:.4f}')

        loss.backward()
        optimizer.step()

    if args.prefix:
        torch.save( model.state_dict(), os.path.join( user.model_directory, 'DGCNN', args.prefix + '_epoch-%d_state.pt' % epoch))
        torch.save( optimizer.state_dict(), os.path.join( user.model_directory, 'DGCNN', args.prefix + '_epoch-%d_optimizer.pt' % epoch))
        
    print(f'Epoch {epoch:03d}, Ntot={n_samples_total:03d}')

    model.eval()
    with torch.no_grad():
        observer_names = dataset.data_config.observers
        obs_features   = {o:np.array([],dtype='float32') for o in observer_names}
        truth          = {'lin':np.array([],dtype='float32')}
        pred           = {'lin':np.array([],dtype='float32')}
        w0             = np.array([],dtype='float32')

        for X, y , observers, batch in training_data:
            points   = torch.Tensor(X['points']).transpose(1,2).to(device)
            features = torch.Tensor(X['features']).transpose(1,2).to(device)
            mask = (points.abs().sum(dim=-1) != 0)
            batch    = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]
            global_features = torch.Tensor(X['global_features'][:,:,0]).to(device) if global_dims>0 else None


            features = (features-features_scaler.mean_.astype("float32"))/features_scaler.scale_.astype("float32")
            points = (points-points_scaler.mean_.astype("float32"))/points_scaler.scale_.astype("float32")
            y_target = (y_target-y_scaler.mean_.astype("float32"))/y_scaler.scale_.astype("float32")
            if global_dims>0:
                global_features = (global_features-global_features_scaler.mean_.astype("float32"))/global_features_scaler.scale_.astype("float32")

            out = model(points[mask], features[mask], batch, global_features)

            # collect data for plotting 
            for o in observer_names:
                obs_features[o] = np.concatenate( (obs_features[o], observers[o]) )
            pred['lin']  = np.concatenate( (pred['lin'], out[:,0].cpu() ))
            truth['lin'] = np.concatenate( (truth['lin'], y['lin_coeff'][:,1] ))
            w0           = np.concatenate( (w0, y['lin_coeff'][:,0] ))
        plot.plot( epoch=epoch, features=[(key, obs_features[key]) for key in observer_names], w0=w0, prediction=pred, truth=truth, 
                   plot_options=dataset.plot_options,
                   plot_directory='DGCNN_%s'%args.prefix + ('_small_%i'%args.small if args.small is not None else '') ,
                   do2D = False, 
                )
