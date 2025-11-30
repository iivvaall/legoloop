# What is it?

A framework for writing neural network training loops based on dependency_injector. 
It allows you to build your custom loop from a modular components

# How to use it?

Let's break down the [MNIST](docs/examples/mnist.ipynb) example.

## data 

First, you typically need to define a dataset class

```python
import legoloop as ll

class MNISTData(ll.SizedDataset):
    def __init__(self, folder, train, aug):
        self.data = MNIST(folder, download=False, train=train)
        self.aug = aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        res = {
            'idx': item,
            'image': self.data.data[item],
            'target': self.data.targets[item],
        }
        return self.aug(res)
```

Note: the dataset consists of records in the form of a dictionary.


## model 

Second step is to define the model and the loss as usual.

```python
from torch import nn
from sklearn import metrics

class SimpleNN(nn.Module):
    def forward(self, batch):
        feat = self.features(batch['image'].unsqueeze(1))
        logits= self.classifier(feat)
        return {
            'logits': logits,
            'probas': nn.Softmax(dim=1)(logits)
        }

class Loss(nn.Module):
    def forward(self, batch, model_out):
        return nn.CrossEntropyLoss()(
            model_out['logits'], batch['target'],
        )

def mnist_metrics(acc):
    logits = acc['logits']
    y_true = acc['target']
    y_pred = np.argmax(logits, axis=1)
    return {
        'accuracy': metrics.accuracy_score(y_pred, y_true) 
    }
```

Note: they also use dicts

## main defs

now it is time to define a loop. Let's split the code into two parts

### the substative part


```python
from dependency_injector import containers, providers
from torch import optim

class MnistDefs(containers.DeclarativeContainer):
    config = providers.Configuration()
    config.from_dict({
        'loaders': {'default': {'batch_size': 256}},
        'lr': 0.001,
        'num_epochs': 8,
        'epoch_size': 60000,
        'eval_samples': 10000
    })
    data_folder = providers.Object('data')
    data = providers.Container(
        ll.Data,
        datasets=providers.Dict(
            train=providers.Singleton(MNISTData, data_folder, train=True, aug=train_aug),
            val=providers.Singleton(MNISTData, data_folder, train=False, aug=train_aug),
            val_real=providers.Singleton(MNISTData, data_folder, train=False, aug=real_aug)
        )
    )
    model = providers.Singleton(SimpleCNN)
    opt = providers.Singleton(
        lambda model, lr: optim.Adam(model.parameters(), lr=lr),
        model, config.lr
    )
    loss = providers.Singleton(Loss)
    metrics = providers.Object(mnist_metrics)
```

### the boilerplate code

```python
class MnistApp(containers.DeclarativeContainer):
    root = providers.Singleton(Path, 'learn')
    defs = providers.Container(MnistDefs)
    counter = providers.Singleton(ll.Counter)
    device = providers.Object('cuda' if torch.cuda.is_available() else 'cpu')
    feed = providers.Singleton(ll.Feed, model=defs.model, device=device)
    descent = providers.Singleton(
        ll.GradientDescent,
        opt=defs.opt, model=defs.model, feed=feed, loss=defs.loss
    )
    epoch = providers.Singleton(
        ll.SimpleEpoch,
        train_iterator=defs.data.loop_iters.provided['train'],
        epoch_size=defs.config.epoch_size
    )
    last = providers.Singleton(ll.LastEpoch, counter, defs.config.num_epochs)
    accum = providers.Factory(
        ll.CompositeAccumulator,
        logits=providers.Factory(ll.NumpyAccumulator),
        target=providers.Factory(ll.NumpyAccumulator),
        idx=providers.Factory(ll.NumpyAccumulator)
    )
    loops_acc = providers.Singleton(
        ll.LoopsAcc,
        loop_iters=defs.data.loop_iters,
        model=defs.model, factory=accum.provider,
        batch_desc=providers.Singleton(ll.DefaultBatchDesc),
        samples_limit=defs.config.eval_samples,
        feed = feed
    )
    loaders_acc = providers.Singleton(
        ll.LoadersAcc,
        loaders=defs.data.loaders,
        model=defs.model, factory=accum.provider,
        batch_desc=providers.Singleton(ll.DefaultBatchDesc),
        feed = feed
    )
    loaders_metrics = providers.Singleton(
        ll.AccumulatorMetrics,
        acc_plugin=loaders_acc,
        method=defs.metrics
    )
    loops_metrics = providers.Singleton(
        ll.AccumulatorMetrics, acc_plugin=loops_acc, method=defs.metrics
    )
    predicts = providers.Singleton(
        ll.CsvLogits,
        path=root.provided.joinpath.call('predicts.csv'),
        acc_plugin=loaders_acc,
        names=[f'logit{num}' for num in range(10)],
        acc_key='val_real',
        index='idx'
    )
    board = providers.Singleton(
        ll.TensorBoard,
        counter=counter,
        acc_metrics=loops_metrics,
        logs_dir=root.provided.joinpath.call('logs')
    )
    model_weights = providers.Singleton(
        ll.SaveModelWeights,
        defs.model,
        root.provided.joinpath.call('weights')
    )

    host = providers.Singleton(
        ll.TrainingHost, plugins=providers.List(
            counter, epoch, descent, last, 
            loops_acc, loops_metrics,
            loaders_acc, loaders_metrics,
            predicts,
            board,
            predicts,
            model_weights        
        )
    )
```

The code details the assembly of a training loop where:

1. A fixed number of epochs is configured
2. Accuracy metrics are set to be calculated at epoch end
3. A TensorBoard logger is wired to the learn/logs directory
4. A callback to save the final model weights is registered


Launching the entire cycle boils down to creating the host object and calling the run method

```python
app = MnistApp()
# for debug 
#app.defs.config.from_dict({'num_epochs': 1, 'epoch_size': 6000, 'eval_samples': 1000})
app.host().run()
```