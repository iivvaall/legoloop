

from legoloop import xor_app, base


def test_board(tmp_path):
    app = xor_app.XorApp(root=tmp_path)
    host = base.TrainingHost(
        plugins=[
            app.counter(), app.epoch(), app.last_epoch(),
            app.loops_acc(), app.loops_metrics(),
            app.board()
        ]
    )
    host.run()
    files = list((tmp_path/'logs').iterdir())
    assert len(files) == 1
    assert 'event' in files[0].name
    assert files[0].stat().st_size > 80