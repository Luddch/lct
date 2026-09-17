import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossBatchHard(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        n = embeddings.size(0)
        labels = labels.view(-1, 1)
        is_pos = labels.eq(labels.t())
        is_neg = ~is_pos

        dist_ap = (dist_mat * is_pos.float()).max(dim=1)[0]
        dist_an_masked = dist_mat.clone()
        dist_an_masked[~is_neg] = float("inf")
        dist_an = dist_an_masked.min(dim=1)[0]

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device="cuda"):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


class ReIDLossTotal(nn.Module):
    def __init__(self, cfg_loss, num_classes, embedding_dim, device):
        super().__init__()
        self.id_criterion = nn.CrossEntropyLoss(label_smoothing=cfg_loss.label_smoothing)
        self.triplet_criterion = TripletLossBatchHard(cfg_loss.triplet_margin)
        self.center_criterion = CenterLoss(num_classes, embedding_dim, device) \
            if cfg_loss.center_weight > 0 else None
        self.cfg = cfg_loss

    def forward(self, outputs, labels):
        logs = {}
        id_loss = self.id_criterion(outputs["logits"], labels)
        triplet_loss = self.triplet_criterion(outputs["global_feat"], labels)

        total = self.cfg.id_weight * id_loss + self.cfg.triplet_weight * triplet_loss
        logs["id_loss"] = id_loss.item()
        logs["triplet_loss"] = triplet_loss.item()

        if self.center_criterion is not None:
            center_loss = self.center_criterion(outputs["bn_feat"], labels)
            total = total + self.cfg.center_weight * center_loss
            logs["center_loss"] = center_loss.item()

        logs["total_loss"] = total.item()
        return total, logs
