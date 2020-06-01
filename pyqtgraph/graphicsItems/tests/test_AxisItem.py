import pyqtgraph as pg

app = pg.mkQApp()

def test_AxisItem_stopAxisAtTick(monkeypatch):
    def test_bottom(p, axisSpec, tickSpecs, textSpecs):
        assert view.mapToView(axisSpec[1]).x() == 0.25
        assert view.mapToView(axisSpec[2]).x() == 0.75

    def test_left(p, axisSpec, tickSpecs, textSpecs):
        assert view.mapToView(axisSpec[1]).y() == 0.875
        assert view.mapToView(axisSpec[2]).y() == 0.125

    plot = pg.PlotWidget()
    view = plot.plotItem.getViewBox()
    bottom = plot.getAxis("bottom")
    bottom.setRange(0, 1)
    bticks = [(0.25, "a"), (0.6, "b"), (0.75, "c")]
    bottom.setTicks([bticks, bticks])
    bottom.setStyle(stopAxisAtTick=(True, True))
    monkeypatch.setattr(bottom, "drawPicture", test_bottom)

    left = plot.getAxis("left")
    lticks = [(0.125, "a"), (0.55, "b"), (0.875, "c")]
    left.setTicks([lticks, lticks])
    left.setRange(0, 1)
    left.setStyle(stopAxisAtTick=(True, True))
    monkeypatch.setattr(left, "drawPicture", test_left)

    plot.show()
    app.processEvents()
    plot.close()


def test_AxisItem_viewUnlink():
    plot = pg.PlotWidget()
    view = plot.plotItem.getViewBox()
    axis = plot.getAxis("bottom")
    assert axis.linkedView() == view
    axis.unlinkFromView()
    assert axis.linkedView() is None


class FakeSignal:

    def __init__(self):
        self.calls = []

    def connect(self, *args, **kwargs):
        self.calls.append('connect')

    def disconnect(self, *args, **kwargs):
        self.calls.append('disconnect')


class FakeView:

    def __init__(self):
        self.sigYRangeChanged = FakeSignal()
        self.sigXRangeChanged = FakeSignal()
        self.sigResized = FakeSignal()


def test_AxisItem_bottomRelink():
    axis = pg.AxisItem('bottom')
    fake_view = FakeView()
    axis.linkToView(fake_view)
    assert axis.linkedView() == fake_view
    assert fake_view.sigYRangeChanged.calls == []
    assert fake_view.sigXRangeChanged.calls == ['connect']
    assert fake_view.sigResized.calls == ['connect']
    axis.unlinkFromView()
    assert fake_view.sigYRangeChanged.calls == []
    assert fake_view.sigXRangeChanged.calls == ['connect', 'disconnect']
    assert fake_view.sigResized.calls == ['connect', 'disconnect']


def test_AxisItem_leftRelink():
    axis = pg.AxisItem('left')
    fake_view = FakeView()
    axis.linkToView(fake_view)
    assert axis.linkedView() == fake_view
    assert fake_view.sigYRangeChanged.calls == ['connect']
    assert fake_view.sigXRangeChanged.calls == []
    assert fake_view.sigResized.calls == ['connect']
    axis.unlinkFromView()
    assert fake_view.sigYRangeChanged.calls == ['connect', 'disconnect']
    assert fake_view.sigXRangeChanged.calls == []
    assert fake_view.sigResized.calls == ['connect', 'disconnect']
