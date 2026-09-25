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