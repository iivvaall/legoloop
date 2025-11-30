import numpy as np
import pandas as pd

from legoloop import accumulators, xor_app
from legoloop import base, train


def test_composite():
    acc = accumulators.CompositeAccumulator(
        lst=accumulators.ListAccumulator(),
        arr=accumulators.NumpyAccumulator(),
    )
    acc.feed({'lst': [1, 2], 'arr': np.array([3, 4])})
    acc.feed({'lst': [3, 4], 'arr': np.array([5, 6])})
    assert acc.result()['lst'] == [1, 2, 3, 4]
    assert acc.result()['arr'].tolist() == [3, 4, 5, 6]


def test_union():
    acc = accumulators.UnionAccumulator(
        accumulators.CompositeAccumulator(
            lst=accumulators.ListAccumulator(),
        ),
        accumulators.CompositeAccumulator(
            arr=accumulators.NumpyAccumulator(),
        )
    )
    acc.feed({'lst': [1, 2], 'arr': np.array([3, 4])})
    acc.feed({'lst': [3, 4], 'arr': np.array([5, 6])})
    assert acc.result()['lst'] == [1, 2, 3, 4]
    assert acc.result()['arr'].tolist() == [3, 4, 5, 6]


def test_loader_acc_and_metrics():
    app = xor_app.XorApp(
        model=xor_app.IdxModel()
    )
    host = base.TrainingHost(
        plugins=[
            app.loaders_acc(),
            app.loaders_metrics(),
            app.epoch(),
            train.OneEpoch()
        ]
    )
    host.run()
    out = app.loaders_acc().accumulators['train'].result()
    assert np.isclose(out['features'], out['logits']).all()
    assert app.loaders_metrics().metrics['train'] == {
        'roc_auc': 0.5, 'precision': 1.0, 'recall': 0.5, 'f1': 0.5
    }


def test_loop_acc_and_metrics():
    app = xor_app.XorApp(
        model=xor_app.IdxModel()
    )
    host = base.TrainingHost(
        plugins=[
            app.loops_acc(),
            app.loops_metrics(),
            app.epoch(),
            train.OneEpoch()
        ]
    )
    host.run()
    out = app.loops_acc().accumulators['train'].result()
    assert np.isclose(out['features'], out['logits']).all()
    assert app.loops_metrics().metrics['train'] == {
        'roc_auc': 0.5, 'precision': 1.0, 'recall': 0.5, 'f1': 0.5
    }


def test_csv_predicts(tmp_path):
    app = xor_app.XorApp(
        model=xor_app.IdxModel(),
        root=tmp_path
    )
    host = base.TrainingHost(
        plugins=[
            app.loops_acc(),
            app.predicts(),
            app.epoch(),
            train.OneEpoch()
        ]
    )
    host.run()
    df = pd.read_csv(tmp_path/'predicts.csv', index_col='idx')
    df = df.sort_index()
    assert df['logit0'].to_list() == [0.0, 1.0, 0.0, 1.0]
    assert sorted(df.columns) == ['logit0', 'logit1']