import math


class BatchSizeScheduler:
    """Линейный рост batch size от start до end за warmup_steps шагов оптимизатора."""

    def __init__(self, start: int, end: int, warmup_steps: int, micro_batch_size: int):
        self.start, self.end = start, end
        self.warmup_steps = warmup_steps
        self.micro = micro_batch_size

    def batch_size(self, step: int) -> int:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return self.end
        bs = self.start + (self.end - self.start) * step / self.warmup_steps
        return max(self.micro, int(bs) // self.micro * self.micro)

    def accumulation_steps(self, step: int) -> int:
        return self.batch_size(step) // self.micro


def cosine(progress: float, start: float, end: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))