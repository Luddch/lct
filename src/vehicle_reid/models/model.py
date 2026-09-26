import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        # Эквивалент weight_norm с g=1: нормируем строки весов в forward
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)

    def forward(self, x):
        x = self.mlp(x)
        # последний слой в fp32: логиты делятся на маленькую температуру (0.04),
        # ошибки bf16 здесь заметно шумят
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.normalize(x.float(), dim=-1)
            w = F.normalize(self.last_layer.weight.float(), dim=-1)
            return F.linear(x, w)


class MultiCropWrapper(nn.Module):
    """Прогоняет кропы разных разрешений через бэкбон отдельно, голову — один раз."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops):
        if isinstance(crops, torch.Tensor):
            crops = [crops]
        feats = torch.cat([self.backbone(c) for c in crops], dim=0)
        return self.head(feats)

# --------------------------------------------------------- ReID-модель (стадия 2)


class GeM(nn.Module):
    """Generalized-mean pooling по токенам ViT (или по H*W у CNN)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))) if learnable else float(p)
        self.eps = eps

    def forward(self, tokens):           # [B, N, C]
        p = self.p if isinstance(self.p, float) else self.p.clamp(min=1.0)
        x = tokens.clamp(min=self.eps).pow(p)
        return x.mean(dim=1).pow(1.0 / p)


class VehicleReIDModel(nn.Module):
    """Бэкбон (ViT из timm, обычно дообученный DINO) + BNNeck + опциональный ArcFace.

    Правило BoT/TransReID: contrastive/triplet считается ДО BNNeck (feat_t),
    классификация — ПОСЛЕ (feat_bn), на инференсе берётся feat_bn.
    """

    def __init__(self, backbone_name: str, pretrained: bool = True, embedding_dim: int = 0,
                 num_classes: int = 0, pooling: str = "cls", neck: str = "bnneck",
                 drop_path_rate: float = 0.1, grad_checkpointing: bool = False,
                 arcface: bool = True, arcface_scale: float = 30.0, arcface_margin: float = 0.30,
                 img_size=None):
        super().__init__()
        import timm

        kwargs = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate,
                      dynamic_img_size=True)
        if img_size is not None:
            kwargs["img_size"] = tuple(img_size)
        self.backbone = timm.create_model(backbone_name, **kwargs)
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        self.pooling_type = pooling
        feat_dim = self.backbone.num_features
        self.gem = GeM() if pooling == "gem" else None

        self.reduction = (
            nn.Linear(feat_dim, embedding_dim, bias=False)
            if embedding_dim and embedding_dim != feat_dim else nn.Identity()
        )
        self.embedding_dim = embedding_dim if isinstance(self.reduction, nn.Linear) else feat_dim

        if neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.embedding_dim)
            self.bottleneck.bias.requires_grad_(False)   # BNNeck: bias отключён
            nn.init.constant_(self.bottleneck.weight, 1.0)
            nn.init.constant_(self.bottleneck.bias, 0.0)
        else:
            self.bottleneck = nn.Identity()

        self.classifier = None
        if num_classes > 0:
            if arcface:
                from src.vehicle_reid.losses.losses import ArcFaceHead
                self.classifier = ArcFaceHead(self.embedding_dim, num_classes,
                                              scale=arcface_scale, margin=arcface_margin)
            else:
                self.classifier = nn.Linear(self.embedding_dim, num_classes, bias=False)
                nn.init.normal_(self.classifier.weight, std=0.001)

    # ------------------------------------------------------------------ utils
    def _forward_features(self, x):
        if self.pooling_type == "gem" and hasattr(self.backbone, "forward_features"):
            tokens = self.backbone.forward_features(x)          # [B, N, C]
            if tokens.ndim == 4:                                # CNN: [B, C, H, W]
                tokens = tokens.flatten(2).transpose(1, 2)
            prefix = getattr(self.backbone, "num_prefix_tokens", 0)
            patch_tokens = tokens[:, prefix:] if prefix else tokens
            return self.gem(patch_tokens)
        return self.backbone(x)                                 # pooled CLS

    def forward(self, x, targets=None):
        feat_t = self.reduction(self._forward_features(x))      # для contrastive
        feat_bn = self.bottleneck(feat_t)                       # для классификации/инференса
        if self.classifier is None or not self.training:
            return {"feat_t": feat_t, "feat_bn": feat_bn, "logits": None}
        logits = (self.classifier(feat_bn, targets)
                  if hasattr(self.classifier, "margin") else self.classifier(feat_bn))
        return {"feat_t": feat_t, "feat_bn": feat_bn, "logits": logits}

    @torch.no_grad()
    def get_embedding(self, x, normalize: bool = True):
        was_training = self.training
        self.eval()
        feat = self.bottleneck(self.reduction(self._forward_features(x)))
        if normalize:
            feat = F.normalize(feat.float(), dim=-1)
        if was_training:
            self.train()
        return feat

    # ------------------------------------------------------- загрузка DINO
    def load_backbone_weights(self, path, strict: bool = False) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("teacher_backbone", "backbone", "model", "state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        state = {k.replace("backbone.", "", 1) if k.startswith("backbone.") else k: v
                 for k, v in state.items()}
        own = self.backbone.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        skipped = sorted(set(state) - set(compatible))
        missing, _ = self.backbone.load_state_dict(compatible, strict=strict)
        return {"loaded": len(compatible), "skipped": skipped, "missing": list(missing)}

    def freeze_blocks(self, n_first: int):
        """Заморозка patch-embed и первых n блоков — главный рычаг экономии VRAM."""
        if n_first <= 0:
            return 0
        frozen = 0
        for name in ("patch_embed", "cls_token", "pos_embed"):
            module = getattr(self.backbone, name, None)
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad_(False)
                    frozen += 1
            elif isinstance(module, nn.Parameter):
                module.requires_grad_(False)
                frozen += 1
        blocks = getattr(self.backbone, "blocks", [])
        for blk in list(blocks)[:n_first]:
            for p in blk.parameters():
                p.requires_grad_(False)
                frozen += 1
        return frozen
