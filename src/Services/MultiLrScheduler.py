import torch


class MultiLrScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def load_state_dict(self, state_dict_list):
        if len(state_dict_list) != len(self.schedulers):
            raise ValueError("len(state_dict_list) != len(self.schedulers)")
        for scheduler, state in zip(self.schedulers, state_dict_list):
            scheduler.load_state_dict(state)

    def get_last_lr(self):
        lrs = []
        for scheduler in self.schedulers:
            lrs.extend(scheduler.get_last_lr())
        return lrs