from pathlib import Path

import torch
from torch import nn

from legoloop import base

class ModelSerializer():
    def save_model(self, model, path):
        raise NotImplementedError()

    def load_model(self, model, path):
        raise NotImplementedError()


class SimpleModelSerializer(ModelSerializer):
    def __init__(self, device):
        self.device = device

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        dct = torch.load(path, map_location=self.device)
        model.load_state_dict(dct)


class SaveModel(base.TrainingPlugin):
    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device

    def save_checkpoint(self, folder: Path):
        torch.save(self.model.state_dict(), folder/'model')

    def load_checkpoint(self, folder: Path):
        state = torch.load(folder/'model', map_location=self.device)
        self.model.load_state_dict(state)