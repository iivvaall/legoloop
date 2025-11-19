

from legoloop import xor_app, base, utility


def test_descent():
    app = xor_app.XorApp()
    host = base.TrainingHost(
        plugins=[app.counter(), app.epoch(), app.descent(), app.last_epoch()]
    )
    host.run()
