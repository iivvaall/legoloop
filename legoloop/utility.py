import dataclasses

import torch

from legoloop import base


class BatchDesc():
    def batch_size(self, batch):
        raise NotImplementedError()


class DefaultBatchDesc(BatchDesc):
    def batch_size(self, batch):
        if torch.is_tensor(batch):
            return len(batch)
        if isinstance(batch, dict):
            sizes = {
                len(value)
                for value in batch.values()
            }
            if len(sizes) != 1:
                raise ValueError('umbiguous or undefined batch size')
            return list(sizes)[0]
        if isinstance(batch, (list, tuple)):
            sizes = {
                len(value)
                for value in batch
            }
            if len(sizes) != 1:
                raise ValueError('umbiguous or undefined batch size')
            return list(sizes)[0]
        raise ValueError('unknown batch format')


class Counter(base.TrainingPlugin):
    @dataclasses.dataclass
    class State:
        epochs: int = 0
        current_epoch: int = 0
        global_batches: int = 0
        global_samples: int = 0
        epoch_batches: int = 0
        epoch_samples: int  = 0

    def __init__(self, batch_desc: BatchDesc = DefaultBatchDesc()):
        super().__init__()
        self.batch_desc = batch_desc

    def epoch_start(self):
        self.state.current_epoch += 1
        self.state.epoch_batches = 0
        self.state.epoch_samples = 0

    def epoch_end(self):
        self.state.epochs = self.state.current_epoch

    def batch_start(self, batch):
        self.state.global_samples += self.batch_desc.batch_size(batch)
        self.state.epoch_samples += self.batch_desc.batch_size(batch)
        self.state.global_batches += 1
        self.state.epoch_batches += 1


class StateHistory(base.TrainingPlugin):

    def __init__(self):
        super().__init__()
        self.by_epoch = []
        self.by_batch = []

    def epoch_end(self):
        self.by_epoch.append(self.host.state())

    def batch(self, batch):
        self.by_batch.append(self.host.state())

    def get_batch_history(self, key):
        return [
            record[key] for record in self.by_batch
        ]

    def get_epoch_history(self, key):
        return [
            record[key] for record in self.by_epoch
        ]