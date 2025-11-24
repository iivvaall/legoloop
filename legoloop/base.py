import enum
import json
import dataclasses
from typing import Optional
from pathlib import Path

from dependency_injector import providers, containers

from legoloop.misc import path_join

class Layout(containers.DeclarativeContainer):
    logs_folder = providers.Dependency()
    output_folder = providers.Dependency()


class SimpleLayout(Layout):
    train_root = providers.Dependency()
    logs_folder = providers.Singleton(path_join,  train_root, 'logs')
    output_folder = providers.Singleton(path_join, train_root, 'output')


class NamedLayout(Layout):
    train_root = providers.Dependency()
    name = providers.Dependency()
    logs_folder = providers.Singleton(path_join,  train_root, 'logs', name)
    output_folder = providers.Singleton(path_join, train_root, 'output', name)


class PluginOutput():
    pass


class ShouldStop(PluginOutput):
    pass


class Stage(enum.StrEnum):
    epoch_start = enum.auto()
    epoch = enum.auto()
    epoch_end = enum.auto()
    train_start = enum.auto()
    train = enum.auto()
    train_end = enum.auto()
    batch_start = enum.auto()
    batch = enum.auto()
    batch_end = enum.auto()


class TrainingPlugin():

    @dataclasses.dataclass()
    class State:
        pass

    def __init__(self):
        self.state = self.State()
        self.host: Optional['TrainingHost'] = None

    def bind(self, host: 'TrainingHost'):
        self.host = host

    def batch_start(self, batch):
        pass

    def batch(self, batch):
        pass

    def batch_end(self, batch):
        pass

    def epoch_start(self):
        pass

    def epoch(self):
        pass

    def epoch_end(self):
        pass

    def train_start(self):
        self.state = self.State()

    def train(self):
        pass

    def train_end(self):
        pass

    def get_state(self):
        return dataclasses.asdict(self.state)

    def set_state_from(self, dct):
        kwargs = {
            field.name: dct[field.name]
            for field in dataclasses.fields(self.State)
            if field.name in dct
        }
        self.state = self.State(**kwargs)

    def save_files(self, folder: Path):
        pass

    def load_files(self, folder: Path):
        pass


class TrainingHost():
    def __init__(self, plugins: list[TrainingPlugin]):
        self.plugins = plugins
        self.should_stop = False
        for plugin in self.plugins:
            plugin.bind(self)

    def one_epoch(self):
        for plugin in self.plugins:
            self._call(plugin.epoch_start)
        for plugin in self.plugins:
            self._call(plugin.epoch)
        for plugin in self.plugins:
            self._call(plugin.epoch_end)

    def run(self):
        self.should_stop = False
        for plugin in self.plugins:
            self._call(plugin.train_start)
        while not self.should_stop:
            self.one_epoch()
        for plugin in self.plugins:
            self._call(plugin.train_end)

    def _call(self, method, *args, **kwargs):
        out = method(*args, **kwargs)
        if isinstance(out, ShouldStop):
            self.should_stop = True

    def state(self):
        res = {}
        for plugin in self.plugins:
            for key, value in plugin.get_state().items():
                if key in res:
                    raise ValueError('duplicate state key %s', key)
                res[key] = value
        return res

    def save_checkpoint(self, folder):
        folder.mkdir(exist_ok=True, parents=True)
        with open(folder / 'state.json', 'w') as fh:
            json.dump(self.state(), fh, indent=4)
        for plugin in self.plugins:
            plugin.save_files(folder)

    def load_checkpoint(self, folder):
        with open(folder / 'state.json') as fh:
            state = json.load(fh)
        for plugin in self.plugins:
            plugin.set_state_from(state)
        for plugin in self.plugins:
            plugin.load_files(folder)


class EpochCheckpoints(TrainingPlugin):

    @dataclasses.dataclass()
    class State:
        checkpoint_number: int = 0

    state: State

    def __init__(self, folder, digits=3):
        super().__init__()
        self.folder = folder
        self.digits = digits

    def _name(self, number):
        res = str(number)
        return '0'* (self.digits - len(res)) + res

    def epoch_end(self):
        self.state.checkpoint_number += 1
        folder = self.folder/self._name(self.state.checkpoint_number)
        self.host.save_checkpoint(folder)

    def load(self, number):
        self.host.load_checkpoint(
            self.folder/self._name(number)
        )