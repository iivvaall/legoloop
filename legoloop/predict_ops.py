from pathlib import Path
import pandas as pd

import torch

class Predictor():
    def make_predicts(self, target_path: Path):
        pass

class LoadersCSVPredictor(Predictor):
    def __init__(self, model, loaders, feed, accumulator_factory, samples_limit):
        self.model = model
        self.loaders = loaders
        self.feed = feed
        self.accumulator_factory = accumulator_factory
        self.samples_limit = samples_limit

    def make_predicts(self, target_path: Path):
        self.model.eval()
        with torch.no_grad():
            for loader_name, loader in self.loaders.items():
                acc = self.accumulator_factory()
                num_samples = 0
                for batch in loader:
                    model_out = self.feed.prediction_feed(batch)
                    acc.feed(model_out)
                    num_samples += self.feed.batch_size(batch)
                    if self.samples_limit is not None and num_samples > self.samples_limit:
                        break
                res: pd.DataFrame
                res = acc.result()
                res.to_csv(target_path / f'{loader_name}.csv')
