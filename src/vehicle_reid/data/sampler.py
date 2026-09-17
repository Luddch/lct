import random
from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """P identities x K instances per batch."""

    def __init__(self, dataset, p: int, k: int):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.index_by_pid = defaultdict(list)
        for idx in range(len(dataset)):
            pid = dataset.df.iloc[idx]["vehicle_id"]
            self.index_by_pid[pid].append(idx)
        self.pids = list(self.index_by_pid.keys())
        self.length = (len(self.pids) // self.p) * self.p * self.k

    def __iter__(self):
        pids = self.pids.copy()
        random.shuffle(pids)
        batches = []
        for i in range(0, len(pids) - self.p + 1, self.p):
            batch = []
            for pid in pids[i:i + self.p]:
                idxs = self.index_by_pid[pid]
                if len(idxs) >= self.k:
                    chosen = random.sample(idxs, self.k)
                else:
                    chosen = list(np.random.choice(idxs, self.k, replace=True))
                batch.extend(chosen)
            batches.append(batch)
        random.shuffle(batches)
        flat = [i for b in batches for i in b]
        return iter(flat)

    def __len__(self):
        return self.length
