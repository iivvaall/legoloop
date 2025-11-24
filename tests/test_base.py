from legoloop import base, xor_app


def test_checkpoints(tmp_path):
    app = xor_app.XorApp(
        root=tmp_path
    )
    host = base.TrainingHost(
        plugins=[
            app.counter(),
            app.epoch(),
            app.descent(),
            app.model_weights(),
            app.last_epoch(),
            app.anneal(),
            app.checkpoints(),
        ]
    )
    host.run()
    assert app.counter().state.epochs == 3
    assert round(app.descent().opt.param_groups[0]['lr'], 5) == round(0.01 / 4, 5)
    app.checkpoints().load(1)
    assert app.counter().state.epochs == 1
    assert round(app.descent().opt.param_groups[0]['lr'], 5) == round(0.01 / 2, 5)