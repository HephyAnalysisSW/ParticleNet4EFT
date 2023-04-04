import warnings

warnings.filterwarnings("ignore")
import os.path as osp
import math

import torch
#import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, global_max_pool
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

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
        return self.mlp(tmp)

from torch_geometric.nn import knn_graph

class DynamicEdgeConv(EdgeConv):
    def __init__(self, mlp, k, aggr):
        super().__init__(mlp, aggr)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)

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

size     = 16
k        = 20

conv_params = [ 
#    (k, [size, size, size]),
#    (k, [2*size, 2*size])
]

fusion_output = None#8*size

norm_kwargs={'track_running_stats':False}

num_classes = 1
conv_dim = 16
#params           = [ (256, 0.01), (256, 0.01 ) ]
global_params    = ( [4*size, 4*size], 0.0 ) 
combined_params  = ( [4*size, 4*size], 0.0 ) 

import dataset
feature_dims = len(dataset.data_config.input_dicts['features']) if 'features' in dataset.data_config.input_dicts else 0
point_dims = len(dataset.data_config.input_dicts['points']) if 'points' in dataset.data_config.input_dicts else 0
global_dims  = len(dataset.data_config.inputs['global_features']['vars'])

class Net(torch.nn.Module):
    def __init__(self, out_channels=1, aggr='max', global_dims=0):
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
        
        self.dEC = []
        dEC_out_chn_total = 0
        dEC_out_chn       = 0
        for idx, (k, conv_param ) in enumerate(conv_params):
            dEC_out_chn_total+=conv_param[-1]
            if idx==0:
                self.dEC.append( DynamicEdgeConv(MLP([2 * point_dims]+conv_param), k, aggr, norm_kwargs=norm_kwargs) )
            else:
                self.dEC.append( DynamicEdgeConv(MLP([2*conv_params[idx-1][1][-1]]+conv_param), k, aggr, norm_kwargs=norm_kwargs) )
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

    def forward(self, points, features, mask, global_features):
    
        #if self.feature_conv:
        #    x = self.feature_conv(features) 
        #else:
        #    x = features 
        #x1 = self.dEC1(points, batch)
        #x2 = self.dEC2(x1, batch)
        #x = self.lin1(torch.cat([x1, x2], dim=1))

        if len(self.dEC)>0:
            points = points[mask,:]
            batch = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]

            outputs = []
            for idx, dEC in enumerate(self.dEC):
                x = dEC( points if idx==0 else x, batch)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(out_channels=1, global_dims=global_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
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


for epoch in range(500):

    model.train()

    total_loss = 0
    for X, y , _, batch in dataset.dataset:
        points   = torch.Tensor(X['points']).transpose(1,2).to(device)
        features = torch.Tensor(X['features']).to(device)
        mask = (points.abs().sum(dim=-1) != 0)

        global_features = torch.Tensor(X['global_features'][:,:,0]).to(device)

        optimizer.zero_grad()

        out = model(points, features, mask, global_features)

        y = torch.Tensor(y['lin_coeff']).to(device)
        #loss = y[:,0]*(out[:,0] - y[:,1])**2
        loss = y[:,0]*criterion(out[:,0], y[:,1])
        loss = loss.sum()
        loss.backward()
        total_loss += loss.item() 
        optimizer.step()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
#    scheduler.step()
#    loss = total_loss / len(train_dataset)
#
#    model.eval()
#
#    correct = 0
#    for test_data in test_loader:
#        test_data = test_data.to(device)
#        with torch.no_grad():
#            pred = model(test_data).max(dim=1)[1]
#        correct += pred.eq(test_data.y).sum().item()
#    test_acc = correct / len(test_loader.dataset)
#    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')

    #assert False, "" 
