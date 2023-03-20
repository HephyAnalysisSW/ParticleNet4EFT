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


class FusionBlock(nn.Module):
    def __init__(self, in_chn, out_chn) -> None:
        super().__init__()
        self.fusion_block = nn.Sequential(
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU(),
            )
        self.out_chn = out_chn

    def forward(self, x):
        return self.fusion_block(x)


class DgnnBlock(nn.Module):
    def __init__(self, input_dims, conv_params, use_fts_bn, use_fusion, for_inference) -> None:
        super().__init__()
        # fts batchnorm
        self.use_fts_bn = use_fts_bn
        self.bn_fts = nn.BatchNorm1d(input_dims) if self.use_fts_bn else None
        # EdgeConvs
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))
        # Fusion block
        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)         # enforce min 128 max 1024 out ch for fusion layer, use multiples of 128
            self.fusion_block = FusionBlock(in_chn, out_chn)
            self.out_chn = out_chn
        else:
            self.out_chn = conv_params[-1][1][-1]
        
    def forward(self, pts, fts, mask=None):
        if mask is None:
            mask = (fts.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)

        pts *= mask
        fts *= mask

        if self.use_fts_bn:
            fts = self.bn_fts(fts) * mask

        coord_shift = (mask == 0) * 1e9
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (pts if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        return fts
    

class FullyConnectedBlock(nn.Module):
    def __init__(self,
                 in_chn: int,
                 layer_params: list[tuple[int,float]] = None,
                 out_chn: tuple[int,float] | int = None,            # maybe change to only int
                 input_transform: nn.Module = None) -> None:
        super().__init__()
        self.layer_params = layer_params if layer_params is not None else []
        if type(out_chn) is tuple: self.layer_params.append(out_chn)
        if type(out_chn) is int: self.layer_params.append((out_chn, 0.0))           # set dropout to zero if out chn given as int
        self.out_chn = self.layer_params[-1][0] if self.layer_params else in_chn

        layers = []
        ch_params = [(in_chn, None)] + self.layer_params
        for (in_ch,_), (out_ch, drop_rate) in zip(ch_params[:-1], ch_params[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(drop_rate)
            ))
        if input_transform is not None:
            layers.insert(0, input_transform)
        
        self.fc_block = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_block(input)


class GlobalAveragePooling(nn.Module):
    def __init__(self, use_counts) -> None:
        super().__init__()
        self.use_counts = use_counts

    def forward(self, input, mask=None):
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1
            return input.sum(dim=-1) / counts
        else:
            return input.mean(dim=-1)


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 global_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 pnet_fc_params=[(128, 0.1)],
                 freeze_pnet: bool = False,
                 globals_fc_params=[(128, 0.1)],
                 freeze_global_fc: bool = False,
                 joined_fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        assert not (freeze_pnet and freeze_global_fc), "Can only freeze either gnn OR global dnn part."

        # Dgnn block
        self.use_fts_bn = use_fts_bn
        self.use_fusion = use_fusion
        self.for_inference = for_inference
        self.dgnn_block = DgnnBlock(input_dims, conv_params, self.use_fts_bn, self.use_fusion, self.for_inference)

        # global averaging layer
        self.use_counts = use_counts
        self.global_average_pooling = GlobalAveragePooling(self.use_counts)

        # ParticleNet fully connected layers
        self.freeze_pnet = freeze_pnet
        self.pnet_fc = FullyConnectedBlock(self.dgnn_block.out_chn, pnet_fc_params)

        # freeze all the layers belonging to the gnn part of ParticleNet
        if self.freeze_pnet:
            self.dgnn_block.requires_grad_(False)
            self.global_average_pooling.requires_grad_(False)
            self.pnet_fc.requires_grad_(False)

        # Global features fully connected layers
        self.freeze_global_fc = freeze_global_fc
        self.globals_fc = FullyConnectedBlock(global_dims, globals_fc_params,
                                               input_transform=nn.Flatten(start_dim=-2))    # (N,C,1) -> (N,D)
        # freeze the global features fc layers
        if self.freeze_global_fc:
            self.globals_fc.requires_grad_(False)

        # joined fully connected layers
        joined_in_ch = self.pnet_fc.out_chn + self.globals_fc.out_chn
        self.joined_fc = FullyConnectedBlock(joined_in_ch, joined_fc_params, num_classes)


    def _forward_global_fc_only(self, global_features):

        # globals_fc
        global_fts = self.globals_fc(global_features)

        # make dummy pnet output for joined fc
        batch_size = global_fts.size()[0]
        dummy_pnet_output = torch.randn(batch_size, self.pnet_fc.out_chn, device=global_features.device)

        # joined_fc
        output = self.joined_fc(
            torch.cat([dummy_pnet_output, global_fts], dim=-1)
            )
        
        return output


    def _forward_pnet_only(self, points, features, global_features, mask=None):

         # Dgnn layers including fts_bn and masking
        particle_fts = self.dgnn_block(points, features, mask)

        # global average pooling
        particle_fts = self.global_average_pooling(particle_fts, mask if self.use_counts else None)

        # pnet_fc
        particle_fts = self.pnet_fc(particle_fts)

        # make dummy global fc output
        batch_size = global_features.size()[0]
        dummy_global_fc_output = torch.randn(batch_size, self.globals_fc.out_chn, device=global_features.device)

        # joined_fc
        output = self.joined_fc(
            torch.cat([particle_fts, dummy_global_fc_output], dim=-1)
            )
        
        return output
        

    def forward(self, points, features, global_features, mask=None):

        if self.freeze_pnet:
            return self._forward_global_fc_only(global_features)
        
        if self.freeze_global_fc:
            return self._forward_pnet_only(points, features, global_features, mask)

        # Dgnn layers including fts_bn and masking
        particle_fts = self.dgnn_block(points, features, mask)

        # global average pooling
        particle_fts = self.global_average_pooling(particle_fts, mask if self.use_counts else None)

        # pnet_fc
        particle_fts = self.pnet_fc(particle_fts)

        # globals_fc
        global_fts = self.globals_fc(global_features)

        # joined_fc
        output = self.joined_fc(
            torch.cat([particle_fts, global_fts], dim=-1)
            )

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
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 pnet_fc_params=[(128, 0.1)],
                 freeze_pnet: bool = False,
                 globals_fc_params=[(128, 0.1)],
                 freeze_global_fc: bool = False,
                 joined_fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 constituents_input_dropout=None,
                 global_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.constituents_input_dropout = nn.Dropout(constituents_input_dropout) if constituents_input_dropout else None
        self.global_input_dropout = nn.Dropout(global_input_dropout) if global_input_dropout else None
        self.constituents_conv = FeatureConv(constituents_features_dims, 32)

        self.freeze_pnet = freeze_pnet
        if self.freeze_pnet:
            self.constituents_conv.requires_grad_(False)
        
        self.pn = ParticleNet(input_dims=32,
                              global_dims=global_features_dims,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              pnet_fc_params=pnet_fc_params,
                              freeze_pnet=freeze_pnet,
                              globals_fc_params=globals_fc_params,
                              freeze_global_fc=freeze_global_fc,
                              joined_fc_params=joined_fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, global_features, constituents_points, constituents_features, constituents_mask ):
        if self.freeze_pnet:
            return self.pn(None, None, global_features, None)
        
        if self.constituents_input_dropout:
            constituents_mask = (self.constituents_input_dropout(constituents_mask) != 0).float()
            constituents_points *= constituents_mask
            constituents_features *= constituents_mask

        points = constituents_points
        features = self.constituents_conv(constituents_features * constituents_mask) * constituents_mask
        # torch.cat((self.chh_conv(chh_features * chh_mask) * chh_mask, self.neh_conv(neh_features * neh_mask) * neh_mask, self.el_conv(el_features * el_mask) * el_mask, self.mu_conv(mu_features * mu_mask) * mu_mask, self.ph_conv(ph_features * ph_mask) * ph_mask), dim=2)
        mask = constituents_mask
        return self.pn(points, features, global_features, mask)
