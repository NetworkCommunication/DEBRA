from torch import nn
from .simsiam import SimSiam, projection_MLP
import torch
from .resnet import resnet18, resnet34, resnet50
# from .resnet_lite import resnet18, resnet34, resnet50


def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")
    print(backbone)
    if castrate:
        # backbone.output_dim = backbone.fc.in_features
        backbone.output_dim = 512
        print(backbone.fc.in_features)
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):

    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    else:
        raise NotImplementedError
    return model

def get_prediction(in_channel, out_channel):

    model = nn.Linear(in_features=in_channel, out_features=out_channel)

    models = nn.Sequential(
        # nn.BatchNorm1d(in_channel),
        nn.Linear(in_features=in_channel, out_features=512, bias=False),
        nn.ReLU(),  # 激活函数ReLU
        # nn.Linear(in_features=1024, out_features=1024),
        # nn.ReLU(),  # 激活函数ReLU
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(in_features=512, out_features=out_channel)
    )
    return models

def get_projector(model_cfg):
    model = projection_MLP(512)
    if model_cfg.proj_layers is not None:
        model.set_layers(model_cfg.proj_layers)
    return model




