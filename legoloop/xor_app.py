
from torch import nn, optim
import numpy as np
from dependency_injector import providers, containers

from legoloop import base, data, utility, train, criteria

class XorDataset(data.SizedDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, item):
        f1, f2 = item % 2, (item >> 1) % 2
        return {
            'features': np.array([f1, f2]),
            'target': (f1 + f2) % 2
        }


class XorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, batch):
        return {
            'logits': self.linear(batch['features'].float())
        }


class XorApp(containers.DeclarativeContainer):
    config = providers.Configuration()
    config.from_dict({
        'loaders': {
            'default': {'batch_size': 3}
        },
        'num_epochs': 3,
        'epoch_size': 4
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
        lambda model, lr: optim.Adam(model.parameters(), lr=lr),
        model=model,
        lr=config.lr
    )
    loss  = providers.Singleton(criteria.BatchCrossEntropy)
    feed = providers.Singleton(train.Feed, model, device)
    counter = providers.Singleton(utility.Counter)
    epoch = providers.Singleton(
        train.SimpleEpoch,
        train_iterator=data.loop_iters.provided['train'],
        epoch_size=config.epoch_size,
    )
    descent = providers.Singleton(train.GradientDescent, model, opt, feed, loss)
    last_epoch = providers.Singleton(train.LastEpoch, counter, config.num_epochs)