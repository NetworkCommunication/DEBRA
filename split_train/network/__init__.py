import sys
from .simsiam import SimSiam, projection_MLP, Satellite
import torch
# from .resnet import resnet18, resnet34, resnet50
from .satellite import resnet18

def get_backbone(backbone, enable_layer, castrate=True):
    # backbone = eval(f"{backbone}()")
    backbone_class = getattr(sys.modules[__name__], backbone)
    backbone = backbone_class(enable_layer)

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_model(model_cfg, enable_layer, end_satellite):
    backbone = get_backbone(model_cfg.backbone, enable_layer)
    if end_satellite:
        model = SimSiam(backbone, 512)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    else:
        model = Satellite(backbone, 512)

    return model

def get_satellite_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_satellite_model(model_cfg):
    backbone = get_satellite_backbone(model_cfg.backbone)

    model = SimSiam(backbone, 64)
    if model_cfg.proj_layers is not None:
        model.projector.set_layers(model_cfg.proj_layers)

    return model