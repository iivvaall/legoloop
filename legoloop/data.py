import random

import numpy as np
import torch
import torch.utils.data as tdutils


from dependency_injector import containers, providers


class SizedDataset(tdutils.Dataset):
    def __len__(self):
        raise NotImplementedError()


class Loop(tdutils.IterableDataset):
    def __init__(self, dataset: SizedDataset):
        self.dataset = dataset

    def __iter__(self):
        worker_info = tdutils.get_worker_info()
        while True:
            if worker_info is None:
                idx = range(len(self.dataset))
            else:
                idx = range(
                    worker_info.id,
                    len(self.dataset),
                    worker_info.num_workers
                )
            idx = list(idx)
            random.shuffle(idx)
            for i in idx:
                yield self.dataset[i]


def make_loops(datasets):
    return {
        name: Loop(dataset)
        for name, dataset in datasets.items()
    }


def _get_loader_param(config, dataset_name, key):
    config = config or {}
    eff_config = {'batch_size': 8, 'num_workers': 0, 'shuffle': True}
    eff_config.update(config.get('default', {}))
    eff_config.update(config.get(dataset_name, {}))
    return eff_config[key]


def make_loaders(datasets, config, shuffle=True):
    return {
        dataset_name: tdutils.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=_get_loader_param(config, dataset_name, 'batch_size'),
            num_workers=_get_loader_param(config, dataset_name, 'num_workers'),
            collate_fn=tdutils.default_collate
        )
        for dataset_name, dataset in datasets.items()
    }


def make_iters(loaders):
    return {
        name: iter(loader)
        for name, loader in loaders.items()
    }


def collate_with_pad(batch, pad_columns=('tokens', 'attention_mask')):
    for colname in pad_columns:
        max_len = max(len(datum[colname]) for datum in batch)
        for datum in batch:
            datum[colname] = np.pad(datum[colname], pad_width=(0, max_len - len(datum[colname])))
    return torch.utils.data.default_collate(batch)


class Data(containers.DeclarativeContainer):
    datasets = providers.Dependency()
    config = providers.Configuration()

    collate_fn = providers.Object()

    loaders = providers.Singleton(
        make_loaders, datasets=datasets, config=config
    )

    loops = providers.Singleton(make_loops, datasets)
    loop_loaders = providers.Singleton(
        make_loaders, loops, config, shuffle=False
    )

    loop_iters = providers.Singleton(make_iters, loop_loaders)