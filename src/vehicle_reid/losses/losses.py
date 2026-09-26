import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Ожидает view-major раскладку:
    student_out: [(n_global + n_local) * B, K], сначала все глобальные виды, потом локальные;
    teacher_out: [n_global * B, K]."""

    def __init__(self, out_dim, n_global_crops=2, n_local_crops=6, student_temp=0.1,
                 teacher_temp=0.06, warmup_teacher_temp=0.04, teacher_temp_warmup_steps=0,
                 center_momentum=0.9):
        super().__init__()
        self.n_global = n_global_crops
        self.n_local = n_local_crops
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp_warmup_steps = teacher_temp_warmup_steps
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.last_stats: dict = {}

    def teacher_temp_at(self, step: int) -> float:
        if self.teacher_temp_warmup_steps <= 0 or step >= self.teacher_temp_warmup_steps:
            return self.teacher_temp
        a = step / self.teacher_temp_warmup_steps
        return self.warmup_teacher_temp + a * (self.teacher_temp - self.warmup_teacher_temp)

    def forward(self, student_out, teacher_out, step: int):
        n_views = self.n_global + self.n_local
        assert student_out.shape[0] % n_views == 0
        assert teacher_out.shape[0] * n_views == student_out.shape[0] * self.n_global

        teacher_out = teacher_out.detach().float()
        s_chunks = (student_out.float() / self.student_temp).chunk(n_views)
        temp = self.teacher_temp_at(step)
        t_probs = F.softmax((teacher_out - self.center) / temp, dim=-1)
        t_chunks = t_probs.chunk(self.n_global)

        total, n_terms = 0.0, 0
        for i, t in enumerate(t_chunks):
            for j, s in enumerate(s_chunks):
                if i == j:  # один и тот же вид у учителя и студента не сравниваем
                    continue
                total = total + torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1).mean()
                n_terms += 1
        total = total / n_terms

        with torch.no_grad():
            entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(-1).mean()
            self.last_stats = {"teacher_entropy": entropy, "teacher_temp": temp}

        self.update_center(teacher_out)
        return total

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

class CrossEntropyLabelSmooth(nn.Module):
    """CE со сглаживанием меток — стандарт для ReID (BoT)."""

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.epsilon)


class ArcFaceHead(nn.Module):
    """Косинусный классификатор с угловым отступом (ArcFace).

    Память: O(num_classes * dim) весов и логитов — дешевле triplet-майнинга,
    потому что не требует больших батчей.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0,
                 margin: float = 0.30):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, features, targets=None):
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = F.normalize(features.float(), dim=-1)
            w = F.normalize(self.weight.float(), dim=-1)
            cosine = F.linear(x, w).clamp(-1 + 1e-7, 1 - 1e-7)
            if targets is None:
                return cosine * self.scale
            theta = torch.acos(cosine)
            one_hot = F.one_hot(targets, num_classes=w.shape[0]).to(cosine.dtype)
            target_logit = torch.cos(theta + self.margin)
            # easy-margin: не уводим за пи, иначе градиент разворачивается
            target_logit = torch.where(theta + self.margin < math.pi, target_logit, cosine - self.margin)
            logits = one_hot * target_logit + (1.0 - one_hot) * cosine
            return logits * self.scale


class SupConMemoryLoss(nn.Module):
    """Supervised contrastive loss с memory-очередью (MoCo-style).

    Зачем очередь: число негативов перестаёт зависеть от batch size, поэтому
    осмысленный контрастив получается и при batch=8..16 на ограниченной VRAM —
    в отличие от batch-hard triplet, которому нужен большой P*K батч.

    Позитивы для якоря — все элементы (батч + очередь) с тем же pid.
    """

    def __init__(self, dim: int, queue_size: int = 8192, temperature: float = 0.07,
                 exclude_self: bool = True):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.exclude_self = exclude_self
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, dim), dim=1))
        self.register_buffer("queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys, labels):
        keys = F.normalize(keys.detach().float(), dim=1)
        labels = labels.detach().long()
        n = keys.shape[0]
        if n > self.queue_size:
            keys, labels, n = keys[: self.queue_size], labels[: self.queue_size], self.queue_size
        ptr = int(self.queue_ptr.item())
        idx = (torch.arange(n, device=keys.device) + ptr) % self.queue_size
        self.queue.index_copy_(0, idx, keys.to(self.queue.dtype))
        self.queue_labels.index_copy_(0, idx, labels)
        self.queue_ptr[0] = (ptr + n) % self.queue_size
        self.queue_filled[0] = min(self.queue_size, int(self.queue_filled.item()) + n)

    def forward(self, features, labels, keys=None, key_labels=None):
        """features — эмбеддинги с градиентом [B, D]; keys — то, что кладём в очередь
        (обычно выход momentum-энкодера; если None, используем features.detach())."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            f = F.normalize(features.float(), dim=1)
            labels = labels.long()
            keys_f = f.detach() if keys is None else F.normalize(keys.detach().float(), dim=1)
            key_labels = labels if key_labels is None else key_labels.long()

            filled = int(self.queue_filled.item())
            if filled > 0:
                q = self.queue[:filled].to(f.dtype)
                q_lab = self.queue_labels[:filled]
                contrast = torch.cat([keys_f, q], dim=0)
                contrast_labels = torch.cat([key_labels, q_lab], dim=0)
            else:
                contrast, contrast_labels = keys_f, key_labels

            logits = f @ contrast.T / self.temperature
            pos_mask = (labels[:, None] == contrast_labels[None, :]).float()
            if self.exclude_self:
                # якорь против собственного ключа из того же сэмпла — вырожденная пара
                b = f.shape[0]
                self_mask = torch.zeros_like(pos_mask)
                self_mask[:b, :b] = torch.eye(b, device=f.device)
                pos_mask = pos_mask * (1.0 - self_mask)
                logits = logits - self_mask * 1e4

            n_pos = pos_mask.sum(1)
            valid = n_pos > 0
            if not valid.any():
                self.enqueue(keys_f, key_labels)
                return features.sum() * 0.0

            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss = -(pos_mask * log_prob).sum(1)[valid] / n_pos[valid]
            loss = loss.mean()

            self.enqueue(keys_f, key_labels)
            return loss


class CenterLoss(nn.Module):
    """Опциональная стягивающая регуляризация центров классов (дёшево по памяти)."""

    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, dim) * 0.01)

    def forward(self, features, labels):
        centers = self.centers[labels]
        return ((features.float() - centers) ** 2).sum(dim=1).mean() * 0.5
