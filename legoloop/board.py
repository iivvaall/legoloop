from torch.utils.tensorboard import SummaryWriter

from legoloop import base
from legoloop.accumulators import AccumulatorMetrics
from legoloop.utility import Counter


class TensorBoard(base.TrainingPlugin):
    def __init__(self, counter: Counter, acc_metrics: AccumulatorMetrics, logs_dir):
        super().__init__()
        self.counter = counter
        self.acc_metrics = acc_metrics
        self.logs_dir = logs_dir
        self.writer = SummaryWriter(log_dir=logs_dir)

    def train_start(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def epoch_end(self):
        for key, dct in self.acc_metrics.metrics.items():
            for name, value in dct.items():
                self.writer.add_scalar(
                    tag=f'{name}/key',
                    scalar_value=float(value),
                    global_step=self.counter.state.global_samples
                )
        self.writer.flush()