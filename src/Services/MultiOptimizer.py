import torch


class MultiOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict_list):
        if len(state_dict_list) != len(self.optimizers):
            raise ValueError("len(state_dict_list) != len(self.optimizers)")

        for opt, state in zip(self.optimizers, state_dict_list):
            opt.load_state_dict(state)