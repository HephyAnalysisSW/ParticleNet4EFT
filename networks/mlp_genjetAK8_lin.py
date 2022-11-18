import torch
from torch import nn
from utils.nn.model.ParticleNet import ParticleNetTagger

# create a MLP class
class MLP(nn.Module):

    def __init__(self, input_nfeatures, num_classes, hidden_layers=(), **kwargs):
        super().__init__(**kwargs)
        channels = [input_nfeatures] + list(hidden_layers) + [num_classes]
        layers = []
        for c, c_next in zip(channels[:-1], channels[1:]):
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.Linear(c, c_next, bias=True))
            # layers.append(nn.BatchNorm1d(c_next))
            layers.append(nn.ReLU())
        del layers[-1:]
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.mlp(x)


def get_model(data_config, **kwargs):
    hidden_layers = (300,100,100)
    _, pf_length, pf_feature_dims = data_config.input_shapes['hl_features']
    # print(pf_length, pf_feature_dims)
    input_dims = pf_length * pf_feature_dims
    num_classes = len(data_config.label_value)
    model = MLP(input_dims, num_classes, hidden_layers=hidden_layers)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
    return model, model_info


def get_loss(data_config, **kwargs):
    # return torch.nn.CrossEntropyLoss()
    return torch.nn.MSELoss()
