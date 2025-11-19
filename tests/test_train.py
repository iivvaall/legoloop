

from legoloop import base, train, xor_app


def test_train():
    app = xor_app.XorApp()
    host = base.TrainingHost(
        plugins=[app.counter(), app.epoch(), history, app.last_epoch()]
    )
    host.run()
