import timm
import torch.nn as nn


def build_backbone(name: str, pretrained: bool = True):
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
    feat_dim = model.num_features
    return model, feat_dim
