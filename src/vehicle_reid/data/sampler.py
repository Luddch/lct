import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """P identities x K instances в каждом micro-batch.

    Нужен не для batch-hard майнинга (мы его не используем), а чтобы в батче
    гарантированно были позитивы — SupCon от этого выигрывает. Работает с
    ReIDTrainDataset, у которого есть атрибут labels.
    """

    def __init__(self, dataset, p: int, k: int, seed: int | None = None):
        self.p = p
        self.k = k
        self.rng = random.Random(seed)
        labels = getattr(dataset, "labels", None)
        if labels is None:
            raise AttributeError("Датасет должен иметь атрибут labels")

        self.index_by_pid: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.index_by_pid[int(label)].append(idx)
        self.pids = sorted(self.index_by_pid)
        if len(self.pids) < p:
            raise ValueError(f"ID в датасете ({len(self.pids)}) меньше, чем p={p}")
        self.length = (len(self.pids) // self.p) * self.p * self.k

    def __iter__(self):
        pids = self.pids.copy()
        self.rng.shuffle(pids)
        flat: list[int] = []
        for i in range(0, len(pids) - self.p + 1, self.p):
            for pid in pids[i:i + self.p]:
                idxs = self.index_by_pid[pid]
                if len(idxs) >= self.k:
                    flat.extend(self.rng.sample(idxs, self.k))
                else:
                    flat.extend(list(np.random.choice(idxs, self.k, replace=True)))
        return iter(flat)

    def __len__(self):
        return self.length
