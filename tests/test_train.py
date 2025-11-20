import numpy as np

from legoloop import base, train, xor_app


def test_descent():
    app = xor_app.XorApp(
        model=xor_app.IdxModel()
    )
    assert np.isclose(app.model().linear.weight.data.cpu().numpy().std(ddof=0), 0.5)
    host = base.TrainingHost(
        plugins=[app.counter(), app.epoch(), app.descent(), app.last_epoch() ]
    )
    host.run()
    # weights changed
    assert not np.isclose(app.model().linear.weight.data.cpu().numpy().std(ddof=0), 0.5)