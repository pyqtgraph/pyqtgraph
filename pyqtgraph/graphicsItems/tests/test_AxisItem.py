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
