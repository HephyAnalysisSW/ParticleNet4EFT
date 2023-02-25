import numpy as np
import torch
import torch.nn as nn

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 global_dims,
                 num_classes,
                 conv_params,
                 fc_params,
                 fc_global_params,
                 fc_combined_params,
                 use_fusion=False,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        
        
        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion: # KEINE AHNUNG WAS DAS MACHT UND WARUM IST DAS HARDGECODED??
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        fcs = []
        fcs_global = []
        fcs_combined = []
        
        
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
                
            else:
                in_chn = fc_params[idx - 1][0]

            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        
        for idx, layer_param in enumerate(fc_global_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn_global = global_dims
            else:
                in_chn_global = fc_global_params[idx - 1][0]

            fcs_global.append(nn.Sequential(nn.Linear(in_chn_global, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        
        # if no combined layers are used:
        if fc_combined_params == None: 
            fcs_combined.append(nn.Linear(fc_params[-1][0] + fc_global_params[-1][0], num_classes))
        else:         
            for idx, layer_param in enumerate(fc_combined_params):
                channels, drop_rate = layer_param
                if idx == 0:
                    in_chn_combined = fc_params[-1][0] + fc_global_params[-1][0] + out_chn if self.use_fusion else fc_params[-1][0]  + fc_global_params[-1][0]
                else:
                    in_chn_combined = fc_combined_params[idx - 1][0]

                fcs_combined.append(nn.Sequential(nn.Linear(in_chn_combined, channels), nn.ReLU(), nn.Dropout(drop_rate)))
                                 
            fcs_combined.append(nn.Linear(fc_combined_params[-1][0], num_classes))
            
        self.fc = nn.Sequential(*fcs)
        self.fc_global = nn.Sequential(*fcs_global)
        self.fc_combined = nn.Sequential(*fcs_combined)

        self.for_inference = for_inference

    def forward(self, points, features, global_features, mask=None):
#         
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.use_counts:
            x = fts.sum(dim=-1) / counts  # divide by the real counts
        else:
            x = fts.mean(dim=-1)

        for idx, layer in enumerate(self.fc):            
            x = layer(x)
        for idx, layer in enumerate(self.fc_global): 
            if idx == 0:
                y = layer(global_features[:,:,0]) #add global features
            else:            
                y = layer(y)   
                
        for idx, layer in enumerate(self.fc_combined):            
            if idx == 0:
                x = layer(torch.cat((x, y), dim=1)) # concat global DNN and DNN after GNN
            else:            
                x = layer(x)      
        output = x

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

class ParticleNetTagger(nn.Module):

    def __init__(self,
                 constituents_features_dims,
                 global_features_dims,
                 num_classes,
                 conv_params,
                 fc_params,
                 fc_global_params,
                 fc_combined_params,
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 constituents_input_dropout=None,
                 global_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.constituents_input_dropout = nn.Dropout(constituents_input_dropout) if constituents_input_dropout else None
        self.global_input_dropout = nn.Dropout(global_input_dropout) if global_input_dropout else None
        self.constituents_conv = FeatureConv(constituents_features_dims, 32)
        #self.global_conv = FeatureConv(global_features_dims, 64) #FIXME should be removed!
        self.pn = ParticleNet(input_dims=32,
                              global_dims=global_features_dims,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              fc_global_params=fc_global_params,
                              fc_combined_params=fc_combined_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, global_features, constituents_points, constituents_features, constituents_mask ):
        if self.constituents_input_dropout:
            constituents_mask = (self.constituents_input_dropout(constituents_mask) != 0).float()
            constituents_points *= constituents_mask
            constituents_features *= constituents_mask

        points   = constituents_points #torch.cat((constituents_points, neh_points, el_points, mu_points, ph_points), dim=2)
        features = self.constituents_conv(constituents_features * constituents_mask) * constituents_mask# torch.cat((self.chh_conv(chh_features * chh_mask) * chh_mask, self.neh_conv(neh_features * neh_mask) * neh_mask, self.el_conv(el_features * el_mask) * el_mask, self.mu_conv(mu_features * mu_mask) * mu_mask, self.ph_conv(ph_features * ph_mask) * ph_mask), dim=2)
        global_features =  global_features #self.global_conv(global_features)
        mask = constituents_mask #torch.cat((chh_mask, neh_mask, el_mask, mu_mask, ph_mask), dim=2)
        return self.pn(points, features, global_features, mask)
