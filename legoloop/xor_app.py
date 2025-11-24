import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import numpy as np
from dependency_injector import providers, containers

from legoloop import (
    base, data, utility, train, criteria, accumulators, model_ops, board
)

class XorDataset(data.SizedDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, item):
        f1, f2 = item % 2, (item >> 1) % 2
        return {
            'idx': f'idx{item}',
            'features': np.array([f1, f2]),
            'target': (f1 + f2) % 2
        }


class XorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(2, 2),
        )

    def forward(self, batch):
        logits = self.seq(batch['features'].float())
        return {
            'logits': logits,
            'probas': nn.Softmax(dim=1)(logits)
        }


class IdxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.linear.weight.data = torch.tensor([
            [1.0,  0.0],
            [0.0,  1.0]
        ])
        self.linear.bias.data = torch.tensor([0, 0]).float()

    def forward(self, batch):
        logits = self.linear(batch['features'].float())
        return {
            'logits': logits,
            'probas': nn.Softmax(dim=1)(logits)
        }


class XorApp(containers.DeclarativeContainer):
    root = providers.Dependency()
    config = providers.Configuration()
    config.from_dict({
        'loaders': {
            'default': {'batch_size': 2}
        },
        'num_epochs': 3,
        'epoch_size': 4,
        'lr': 0.1
    })
    data = providers.Container(
        data.Data,
        datasets=providers.Dict(
            train=providers.Singleton(XorDataset)
        ),
        config=config.loaders
    )
    device = providers.Object('cpu')
    model = providers.Singleton(XorModel)
    opt = providers.Singleton(
        lambda model, lr: optim.SGD(model.parameters(), lr=lr),
        model=model,
        lr=config.lr
    )
    loss  = providers.Singleton(criteria.BatchCrossEntropy)
    feed = providers.Singleton(train.Feed, model, device)
    batch_desc = providers.Singleton(utility.DefaultBatchDesc)
    counter = providers.Singleton(utility.Counter)
    epoch = providers.Singleton(
        train.SimpleEpoch,
        train_iterator=data.loop_iters.provided['train'],
        epoch_size=config.epoch_size,
    )


    sheduler = providers.Singleton(
        lr_scheduler.LambdaLR,
        optimizer=opt,
        lr_lambda=lambda epoch: 0.1 / (epoch + 1)
    )
    anneal = providers.Singleton(train.LrSheduler, sheduler)
    accumulator = providers.Factory(
        accumulators.CompositeAccumulator,
        idx=providers.Factory(accumulators.ListAccumulator),
        logits=providers.Factory(accumulators.NumpyAccumulator),
        target=providers.Factory(accumulators.NumpyAccumulator),
        probas=providers.Factory(accumulators.NumpyAccumulator),
        features=providers.Factory(accumulators.NumpyAccumulator),
    )
    loaders_acc = providers.Singleton(
        accumulators.LoadersAcc,
        loaders=data.loaders,
        model=model,
        factory=accumulator.provider,
        batch_desc=batch_desc,
        feed=feed
    )
    loops_acc = providers.Singleton(
        accumulators.LoopsAcc,
        loop_iters=data.loop_iters,
        model=model,
        factory=accumulator.provider,
        batch_desc=batch_desc,
        samples_limit=4,
        feed = feed
    )
    loaders_metrics = providers.Singleton(
        accumulators.AccumulatorMetrics,
        acc_plugin=loaders_acc,
        method=providers.Singleton(accumulators.BinaryMetrics)
    )
    loops_metrics = providers.Singleton(
        accumulators.AccumulatorMetrics,
        acc_plugin=loops_acc,
        method=providers.Singleton(accumulators.BinaryMetrics)
    )
    predicts = providers.Singleton(
        accumulators.CsvLogits,
        path=root.provided.joinpath.call('predicts.csv'),
        acc_plugin=loops_acc,
        names=['logit0', 'logit1'],
        acc_key='train',
        index='idx'
    )
    model_weights = providers.Singleton(
        model_ops.SaveModelWeights,
        model,
        path=root.provided.joinpath.call('model'),
    )
    checkpoints = providers.Singleton(
        base.EpochCheckpoints,
        folder=root.provided.joinpath.call('checkpoints'),
    )
    board = providers.Singleton(
        board.TensorBoard,
        counter=counter,
        acc_metrics=loops_metrics,
        logs_dir=root.provided.joinpath.call('logs'),
    )
    descent = providers.Singleton(train.GradientDescent, model, opt, feed, loss)
    last_epoch = providers.Singleton(train.LastEpoch, counter, config.num_epochs)