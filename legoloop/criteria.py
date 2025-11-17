
from torch import nn


class BatchCrossEntropy(nn.Module):
    def __init__(self, target_col='target', logits_col='logits', **kwargs):
        super().__init__()
        self.logits_col = logits_col
        self.target_col = target_col
        self.nn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, batch, model_out):
        return nn.CrossEntropyLoss()(
            model_out[self.logits_col], batch[self.target_col],
        )