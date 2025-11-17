from torch import nn

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