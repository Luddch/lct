from abc import ABC, abstractmethod

class BatchSizeSchedulerBase(ABC):
    """Scheduler для динамического изменения batch size"""
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
