import json
import pandas as pd
import torch
import numpy as np

from sklearn import metrics

from legoloop import base, utility


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


class UnionAccumulator(Accumulator):
    def __init__(self, *accumulators):
        self.accumulators = accumulators

    def feed(self, batch):
        for accumulator in self.accumulators:
            accumulator.feed(batch)

    def result(self):
        res = {}
        for accumulator in self.accumulators:
            res.update(accumulator.result())
        return res


class AccumulatorPlugin(base.TrainingPlugin):
    def __init__(self, model, when=(base.Stage.epoch_end,)):
        super().__init__()
        self.model = model
        self.accumulator = Accumulator()
        self.when = when

    def _prepare_model(self):
        self.model.eval()

    def epoch_end(self):
        if base.Stage.epoch_end not in self.when:
            return
        self._prepare_model()
        self._accumulate()

    def train_end(self):
        if base.Stage.train_end not in self.when:
            return
        self._prepare_model()
        self._accumulate()

    def _accumulate(self):
        pass


class LoopsAcc(AccumulatorPlugin):
    def __init__(self, model, feed, loop_iters, samples_limit, factory, batch_desc: utility.BatchDesc):
        super().__init__(model)
        self.feed = feed
        self.loop_iters = loop_iters
        self.factory = factory
        self.samples_limit = samples_limit
        self.batch_desc = batch_desc

    def _accumulate(self):
        self.accumulator = CompositeAccumulator(**{
            name: self.factory()
            for name in self.loop_iters.keys()
        })
        for name, loop_iter in self.loop_iters.items():
            num_samples = 0
            while num_samples < self.samples_limit:
                batch = next(loop_iter)
                out = self.feed.validation_feed(batch)
                self.accumulator.feed({name: out})
                num_samples += self.batch_desc.batch_size(batch)


class LoadersAcc(AccumulatorPlugin):
    def __init__(self, model, feed, loaders, factory, batch_desc: utility.BatchDesc):
        super().__init__(model)
        self.feed = feed
        self.loaders = loaders
        self.factory = factory
        self.batch_desc = batch_desc

    def _accumulate(self):
        self.accumulator = CompositeAccumulator(**{
            name: self.factory()
            for name in self.loaders.keys()
        })
        for name, loader in self.loaders.items():
            for batch in loader:
                out = self.feed.validation_feed(batch)
                self.accumulator.feed({name: out})


class CsvLogits(base.TrainingPlugin):
    def __init__(self, path, acc_plugin: AccumulatorPlugin, index, names, acc_key='val', logits='logits'):
        super().__init__()
        self.acc_plugin = acc_plugin
        self.acc_key = acc_key
        self.index = index
        self.names = names
        self.logits = logits
        self.path = path

    def train_end(self):
        acc = self.acc_plugin.accumulator.result()[self.acc_key]
        df = pd.DataFrame(
            index=pd.Index(acc[self.index], name=self.index),
            data=acc[self.logits],
            columns=self.names
        ).sort_index()
        df.to_csv(self.path)


class AccumulatorMetrics(base.TrainingPlugin):
    def __init__(self, acc_plugin: AccumulatorPlugin, method):
        super().__init__()
        self.acc_plugin = acc_plugin
        self.method = method
        self.metrics = {}

    def epoch_end(self):
        self.metrics = {
            name: self.method(data)
            for name, data in self.acc_plugin.accumulator.result().items()
        }

    def save_files(self, folder):
        if not self.metrics:
            return
        with open(folder/'metrics.json', 'w') as fh:
            json.dump(self.metrics, fh)


class BinaryMetrics():
    def __init__(self, probas='probas', target='target', thres=0.5):
        self.probas = probas
        self.target = target
        self.thres = thres

    def __call__(self, acc):
        probas = acc[self.probas]
        y_true = acc[self.target]
        y_pred = probas[:,1] > self.thres
        return {
            'roc_auc': metrics.roc_auc_score(y_true, probas[:,1]),
            'precision': metrics.precision_score(y_true, y_pred),
            'recall': metrics.recall_score(y_true, y_pred),
            'f1': metrics.recall_score(y_true, y_pred),
        }