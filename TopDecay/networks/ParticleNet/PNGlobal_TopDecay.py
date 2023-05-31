import torch
from torch import nn
from utils.nn.model.ParticleNetTopDecay import FullyConnectedBlock, DgnnBlock, GlobalAveragePooling, FeatureConv


class ParticleNetEFT(nn.Module):

    def __init__(self,
                 input_dims,
                 global_dims,
                 num_classes,
                 feature_conv_out_dim: int = 32,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 pnet_fc_params=[(128, 0.1)],
                 freeze_pnet: bool = False,
                 globals_fc_params=[(128, 0.1)],
                 freeze_global_fc: bool = False,
                 joined_fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 **kwargs):
        super().__init__(**kwargs)
        assert not (freeze_pnet and freeze_global_fc), "Can only freeze either gnn OR global dnn part."
        print(locals())
        self.num_classes = num_classes
        # FeatureConv before Dgnn block
        self.feature_conv = FeatureConv(input_dims, feature_conv_out_dim)
        # Dgnn block
        self.use_fts_bn = use_fts_bn
        self.use_fusion = use_fusion
        self.dgnn_block = DgnnBlock(feature_conv_out_dim, conv_params, self.use_fts_bn, self.use_fusion)
        # global averaging layer
        self.use_counts = use_counts
        self.global_average_pooling = GlobalAveragePooling(self.use_counts)
        # ParticleNet fully connected layers
        self.freeze_pnet = freeze_pnet
        self.pnet_fc = FullyConnectedBlock(self.dgnn_block.out_chn, pnet_fc_params, use_bn=False, input_transform=nn.BatchNorm1d(self.dgnn_block.out_chn))
        # freeze all the layers belonging to the gnn part of ParticleNet
        if self.freeze_pnet:
            self.feature_conv.requires_grad_(False)
            self.dgnn_block.requires_grad_(False)
            self.global_average_pooling.requires_grad_(False)
            self.pnet_fc.requires_grad_(False)
        # Global features fully connected layers
        self.freeze_global_fc = freeze_global_fc
        global_fc_input_transform = (nn.Flatten(start_dim=-2), nn.BatchNorm1d(global_dims))
        self.globals_fc = FullyConnectedBlock(global_dims, globals_fc_params, use_bn=False, input_transform=global_fc_input_transform)    # (N,C,1) -> (N,C)
        # freeze the global features fc layers
        if self.freeze_global_fc:
            self.globals_fc.requires_grad_(False)
        # joined fully connected layers
        joined_in_ch = self.pnet_fc.out_chn + self.globals_fc.out_chn
        self.joined_fc = FullyConnectedBlock(joined_in_ch, joined_fc_params, num_classes, use_bn=False, is_output_layer=True)


    ### !!! order from data config is: global_features, constituent_points, eflow_features, mask

    def _forward_global_fc_only(self, global_features):
        # globals_fc
        global_fts = self.globals_fc(global_features)
        # make dummy pnet output for joined fc
        batch_size = global_fts.size()[0]
        dummy_pnet_output = torch.randn(batch_size, self.pnet_fc.out_chn, device=global_features.device)
        # joined_fc
        output = self.joined_fc(
            torch.cat((dummy_pnet_output, global_fts), dim=-1)
            )
        return output


    def _forward_pnet_only(self, global_features, points, features, mask=None):
        # FeatureConv
        features = self.feature_conv(features * mask) * mask
        # Dgnn layers including fts_bn and masking, pass through mask
        particle_fts, mask_through = self.dgnn_block(points, features, mask)
        # global average pooling
        particle_fts = self.global_average_pooling(particle_fts, mask_through if self.use_counts else None)
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
        

    def forward(self, global_features, points, features, mask=None):
        if self.freeze_pnet:
            return self._forward_global_fc_only(global_features)
        if self.freeze_global_fc:
            return self._forward_pnet_only(points, features, global_features, mask)
        # FeatureConv
        features = self.feature_conv(features * mask) * mask
        # Dgnn layers including fts_bn and masking, pass through mask
        particle_fts, mask_through = self.dgnn_block(points, features, mask)
        # global average pooling
        particle_fts = self.global_average_pooling(particle_fts, mask_through if self.use_counts else None)
        # pnet_fc
        particle_fts = self.pnet_fc(particle_fts)
        # globals_fc
        global_fts = self.globals_fc(global_features)
        # joined_fc
        output = self.joined_fc(
            torch.cat((particle_fts, global_fts), dim=-1)
            )
        return output