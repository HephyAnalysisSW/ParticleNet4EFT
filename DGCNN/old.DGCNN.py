import warnings

warnings.filterwarnings("ignore")
import os.path as osp

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

size = 16

class Net(torch.nn.Module):
    def __init__(self, n_feature_dim=2, out_channels=1, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * n_feature_dim, size, size, size]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * size, 2*size]), k, aggr)
        self.lin1 = Linear(3*size, 8*size)

        self.mlp = MLP([8*size, 4*size, 4*size, out_channels], dropout=0.5, norm=None)

    def forward(self, data, batch):
        #pos, batch = data.pos, data.batch
        x1 = self.conv1(data, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        return self.mlp(out)
        #return F.log_softmax(out, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(n_feature_dim=2, out_channels=1, k=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

import dataset


for epoch in range(10):

    model.train()

    total_loss = 0
    for X, y , _, batch in dataset.dataset:
        pts = torch.Tensor(X['points']).transpose(1,2).to(device)
        mask = (pts.abs().sum(dim=-1) != 0)
        pts = pts[mask,:]
        batch = (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]
        y = torch.Tensor(y['lin_coeff']).to(device)
        optimizer.zero_grad()
        out = model(pts, batch)
        loss = y[:,0]*(out - y[:,1])**2
        loss = loss.sum()
        loss.backward()
        total_loss += loss.item() 
        optimizer.step()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
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
#    scheduler.step()

    #assert False, "" 
