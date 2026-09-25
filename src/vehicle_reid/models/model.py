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