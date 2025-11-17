import numpy as np
import torch


class Accumulator():
    def feed(self, batch):
        pass

    def result(self):
        return {}


class CompositeAccumulator(Accumulator):
    def __init__(self, **accumulators):
        self.accumulators = accumulators

    def feed(self, model_out):
        for key, value in model_out.items():
            if key in self.accumulators:
                self.accumulators[key].feed(value)
        for key in self.accumulators.keys():
            assert key in model_out, f'{key} not found'

    def result(self):
        return {
            key: acc.result()
            for key, acc in self.accumulators.items()
        }


class NumpyAccumulator(Accumulator):
    def __init__(self):
        self.lst = []

    def feed(self, model_out):
        if isinstance(model_out, torch.Tensor):
            model_out = model_out.detach().cpu().numpy()
        self.lst.append(model_out)

    def result(self):
        return np.concatenate(self.lst)


class ListAccumulator(Accumulator):
    def __init__(self):
        self.lst = []

    def feed(self, model_out):
        self.lst.extend(model_out)

    def result(self):
        return self.lst


class Validator():
    def validate(self, model):
        raise NotImplementedError()


class LoaderValidator(Validator):
    def __init__(self, model, loaders, feed, accumulator_factory, device, samples_limit):
        self.model = model
        self.device = device
        self.loaders = loaders
        self.accumulator_factory = accumulator_factory
        self.feed = feed
        self.samples_limit = samples_limit

    def validate(self, model):
        res = {}
        model.eval()
        with torch.no_grad():
            for loader_name, loader in self.loaders.items():
                acc = self.accumulator_factory()
                num_samples = 0
                for batch in loader:
                    batch = to_device(batch, self.device)
                    model_out = self.feed.validation_feed(batch)
                    acc.feed(model_out)
                    num_samples += self.feed.batch_size(batch)
                    if self.samples_limit is not None and num_samples > self.samples_limit:
                        break
                for key, value in acc.result().items():
                    res[f'{key}/{loader_name}'] = value
        return res