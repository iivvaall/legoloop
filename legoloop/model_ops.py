from pathlib import Path

import torch
from torch import nn

from legoloop import base


class ModelCheckpoint(base.TrainingPlugin):
    def __init__(self, model: nn.Module, device):
        super().__init__()
        self.model = model
        self.device = device

    def save_files(self, folder: Path):
        torch.save(self.model.state_dict(), folder/'model')

    def load_files(self, folder: Path):
        state = torch.load(folder/'model', map_location=self.device)
        self.model.load_state_dict(state)


class SaveModelWeights(base.TrainingPlugin):
    def __init__(self, model, path):
        super().__init__()
        self.model = model
        self.path = path

    def train_end(self):
        torch.save(self.model.state_dict(), self.path)

    def save_files(self, folder: Path):
        torch.save(self.model.state_dict(), self.path)