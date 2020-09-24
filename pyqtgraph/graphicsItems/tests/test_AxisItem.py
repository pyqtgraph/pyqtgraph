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


def test_AxisItem_tickFont(monkeypatch):
    def collides(textSpecs):
        fontMetrics = pg.Qt.QtGui.QFontMetrics(font)
        for rect, _, text in textSpecs:
            br = fontMetrics.tightBoundingRect(text)
            if rect.height() < br.height() or rect.width() < br.width():
                return True
        return False

    def test_collision(p, axisSpec, tickSpecs, textSpecs):
        assert not collides(textSpecs)

    plot = pg.PlotWidget()
    bottom = plot.getAxis("bottom")
    left = plot.getAxis("left")
    font = bottom.linkedView().font()
    font.setPointSize(25)
    bottom.setStyle(tickFont=font)
    left.setStyle(tickFont=font)
    monkeypatch.setattr(bottom, "drawPicture", test_collision)
    monkeypatch.setattr(left, "drawPicture", test_collision)

    plot.show()
    app.processEvents()
    plot.close()


def test_AxisItem_label_visibility():
    axis = pg.AxisItem('left')
    assert axis.labelText == ''
    assert axis.labelUnits == ''
    assert not axis.label.isVisible()

    axis.setLabel(text='Visible')
    assert axis.label.isVisible()
    assert axis.labelText == 'Visible'
    assert axis.labelUnits == ''

    axis.setLabel(text='')
    assert not axis.label.isVisible()
    assert axis.labelText == ''
    assert axis.labelUnits == ''

    axis.setLabel(units='m')
    assert axis.label.isVisible()
    assert axis.labelText == ''
    assert axis.labelUnits == 'm'

    axis.setLabel(units='')
    assert not axis.label.isVisible()
    assert axis.labelText == ''
    assert axis.labelUnits == ''
    assert not axis.label.isVisible()
