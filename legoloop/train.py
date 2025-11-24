import dataclasses
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler

from legoloop import misc, base, utility


class Feed():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_feed(self, batch):
        misc.to_device(batch, self.device)
        return self.model(batch)

    def validation_feed(self, batch):
        res = misc.to_device(batch.copy(), self.device)
        res.update(self.train_feed(res))
        return res

    def prediction_feed(self, batch):
        return self.validation_feed(batch)


class SimpleEpoch(base.TrainingPlugin):
    def __init__(
            self,
            train_iterator,
            epoch_size: int,
            batch_desc: utility.BatchDesc = utility.DefaultBatchDesc()
    ):
        super().__init__()
        self.epoch_size = epoch_size
        self.train_iterator = train_iterator
        self.batch_desc = batch_desc

    def epoch(self):
        epoch_samples = 0
        while epoch_samples < self.epoch_size:
            batch = next(self.train_iterator)
            for plugin in self.host.plugins:
                plugin.batch_start(batch)
            for plugin in self.host.plugins:
                plugin.batch(batch)
            for plugin in self.host.plugins:
                plugin.batch_end(batch)
            epoch_samples += self.batch_desc.batch_size(batch)


class GradientDescent(base.TrainingPlugin):
    def __init__(self, model, opt, feed: Feed, loss):
        super().__init__()
        self.model = model
        self.feed = feed
        self.opt = opt
        self.loss = loss

    def epoch_start(self):
        self.model.train()

    def epoch_end(self):
        self.model.eval()

    def batch(self, batch):
        self.model.zero_grad()
        model_out = self.feed.train_feed(batch)
        loss = self.loss(batch, model_out)
        loss.mean().backward()
        self.opt.step()

    def save_files(self, folder: Path):
        torch.save(self.opt.state_dict(), folder / 'opt')

    def load_files(self, folder: Path):
        self.opt.load_state_dict(
            torch.load(folder / 'opt')
        )


class LastEpoch(base.TrainingPlugin):
    def __init__(self, counters: utility.Counter, last_epoch):
        super().__init__()
        self.counters = counters
        self.last_epoch = last_epoch
        assert self.last_epoch > 0

    def epoch_end(self):
        if self.counters.state.current_epoch == self.last_epoch:
            return base.ShouldStop()
        return None


class OneEpoch(base.TrainingPlugin):
    def epoch_end(self):
        return base.ShouldStop()


class LrSheduler(base.TrainingPlugin):
    @dataclasses.dataclass
    class State:
        sheduler_step: int = 0

    state: State

    def __init__(self, sheduler: lr_scheduler.LRScheduler):
        super().__init__()
        self.sheduler = sheduler

    def epoch_end(self):
        self.state.sheduler_step += 1
        self.sheduler.step()

    def save_files(self, folder: Path):
        torch.save(
            self.sheduler.state_dict(),
            folder/'lr_sheduler'
        )

    def load_files(self, folder: Path):
        self.sheduler.load_state_dict(
            torch.load(folder / 'lr_sheduler')
        )
