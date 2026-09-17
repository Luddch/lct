import torch
import torch.nn as nn

from src.vehicle_reid.models.backbone import build_backbone
from src.vehicle_reid.models.pooling import GeM
from src.vehicle_reid.models.heads import ArcMarginHead, AttributeHead


class VehicleReIDModel(nn.Module):
    """
    Backbone -> GeM pooling -> Bottleneck (BNNeck) -> embedding
    embedding используется на inference (после BN, L2-нормированный)
    ArcFace-голова используется только на train для ID-классификации.
    Опциональные головы атрибутов.
    """

    def __init__(self, backbone_name, pretrained, embedding_dim,
                 num_classes, pooling="gem",
                 arcface_scale=30.0, arcface_margin=0.3,
                 attribute_num_classes: dict | None = None):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.pool = GeM() if pooling == "gem" else nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)  # BNNeck trick (Luo et al.)

        self.reduce = None
        if embedding_dim != feat_dim:
            self.reduce = nn.Linear(feat_dim, embedding_dim)
            self.bottleneck = nn.BatchNorm1d(embedding_dim)
            self.bottleneck.bias.requires_grad_(False)

        self.arc_head = ArcMarginHead(embedding_dim, num_classes, arcface_scale, arcface_margin)

        self.attribute_heads = nn.ModuleDict()
        if attribute_num_classes:
            for name, n_cls in attribute_num_classes.items():
                self.attribute_heads[name] = AttributeHead(embedding_dim, n_cls)

    def extract_backbone_feat(self, x):
        feat_map = self.backbone.forward_features(x)
        pooled = self.pool(feat_map).flatten(1)
        if self.reduce is not None:
            pooled = self.reduce(pooled)
        return pooled

    def forward(self, x, labels=None):
        global_feat = self.extract_backbone_feat(x)          # для triplet-loss
        bn_feat = self.bottleneck(global_feat)                # для ID-loss / inference

        out = {"global_feat": global_feat, "bn_feat": bn_feat}

        if labels is not None:
            out["logits"] = self.arc_head(bn_feat, labels)
            for name, head in self.attribute_heads.items():
                out[f"attr_logits_{name}"] = head(bn_feat)
        return out

    @torch.no_grad()
    def get_embedding(self, x):
        """Inference: нормированный эмбеддинг для векторного поиска."""
        global_feat = self.extract_backbone_feat(x)
        bn_feat = self.bottleneck(global_feat)
        emb = torch.nn.functional.normalize(bn_feat, dim=1)
        return emb
