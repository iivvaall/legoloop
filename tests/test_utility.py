

from legoloop import xor_app, base, utility


def test_counters():
    app = xor_app.XorApp()
    app.config.from_dict({
        'loaders': {
            'default': {'batch_size': 3}
        },
    })
    history = utility.StateHistory()
    host = base.TrainingHost(
        plugins=[app.counter(), history, app.epoch(), app.last_epoch()]
    )
    host.run()
    assert history.get_epoch_history('global_samples') == [6, 12, 18]
    assert history.get_epoch_history('global_batches') == [2, 4, 6]
    assert history.get_batch_history('epoch_samples') == [3, 6] * 3
    assert history.get_batch_history('epoch_batches') == [1, 2] * 3