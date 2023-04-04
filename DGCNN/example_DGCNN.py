import warnings

warnings.filterwarnings("ignore")
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
#from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from torch_geometric.nn import MLP, global_max_pool

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/ModelNet10')
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         num_workers=6)

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
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, size, size, size]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * size, 2*size]), k, aggr)
        self.lin1 = Linear(3*size, 8*size)

        self.mlp = MLP([8*size, 4*size, 4*size, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(1, 201):

    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        assert False, ""
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    loss = total_loss / len(train_dataset)

    model.eval()

    correct = 0
    for test_data in test_loader:
        test_data = test_data.to(device)
        with torch.no_grad():
            pred = model(test_data).max(dim=1)[1]
        correct += pred.eq(test_data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    scheduler.step()

    #assert False, "" 
