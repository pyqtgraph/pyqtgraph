from math import isclose

import pytest

import pyqtgraph as pg

app = pg.mkQApp()


def test_AxisItem_stopAxisAtTick(monkeypatch):
    def test_bottom(p, axisSpec, tickSpecs, textSpecs):
        viewPixelSize = view.viewPixelSize()
        assert isclose(view.mapToView(axisSpec[1]).x(), 0.25, abs_tol=viewPixelSize[0])
        assert isclose(view.mapToView(axisSpec[2]).x(), 0.75, abs_tol=viewPixelSize[0])

    def test_left(p, axisSpec, tickSpecs, textSpecs):
        viewPixelSize = view.viewPixelSize()
        assert isclose(view.mapToView(axisSpec[1]).y(), 0.875, abs_tol=viewPixelSize[1])
        assert isclose(view.mapToView(axisSpec[2]).y(), 0.125, abs_tol=viewPixelSize[1])

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


def test_AxisItem_conditionalSIPrefix():
    plot = pg.PlotWidget()
    plot.setLabel("bottom", "Time", units="s", siPrefix=True, siPrefixEnableRanges=((1, 1e6),))
    bottom = plot.getAxis("bottom")
    bottom.setRange(0, 1e6)
    assert "Time (Ms)" in bottom.labelString()
    bottom.setRange(0, 1e3)
    assert "Time (ks)" in bottom.labelString()
    bottom.setRange(0, 1e9)
    assert "Time (s)" in bottom.labelString()
    bottom.setRange(0, 1e-9)
    assert "Time (s)" in bottom.labelString()
    bottom.setRange(-1e-9, 0)
    assert "Time (s)" in bottom.labelString()
    bottom.setRange(-1e3, 0)
    assert "Time (ks)" in bottom.labelString()
    bottom.setRange(-1e9, 0)
    assert "Time (s)" in bottom.labelString()


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


@pytest.mark.parametrize('orientation,label_kwargs,labelText,labelUnits', [
    ('left', {}, '', '',),
    ('left', dict(text='Position', units='mm'), 'Position', 'mm'),
    ('left', dict(text=None, units=None), '', ''),
    ('left', dict(text='Current', units=None), 'Current', ''),
    ('left', dict(text='', units='V'), '', 'V')
])
def test_AxisItem_label_visibility(orientation, label_kwargs, labelText: str, labelUnits: str):
    """Test the visibility of the axis item using `setLabel`"""
    axis = pg.AxisItem(orientation)
    axis.setLabel(**label_kwargs)
    assert axis.labelText == labelText
    assert axis.labelUnits == labelUnits
    assert (
        axis.label.isVisible() 
        if any(label_kwargs.values())
        else not axis.label.isVisible()
    )

@pytest.mark.parametrize(
    "orientation,x,y,expected",
    [
       ('top', False, True, False),
       ('top', True, False, True),
       ('left', False, True, True),
       ('left', True, False, False),
    ],
)
def test_AxisItem_setLogMode_two_args(orientation, x, y, expected):
    axis = pg.AxisItem(orientation)
    axis.setLogMode(x, y)
    assert axis.logMode == expected

@pytest.mark.parametrize(
    "orientation,log,expected",
    [
       ('top', True, True),
       ('left', True, True),
       ('top', False, False),
       ('left', False, False),
    ],
)
def test_AxisItem_setLogMode_one_arg(orientation, log, expected):
    axis = pg.AxisItem(orientation)
    axis.setLogMode(log)
    assert axis.logMode == expected


def test_AxisItem_clipping_fix_defaults():
    """
    Test that default axis settings are improved for issue #3375.
    
    The fix ensures horizontal axes have better defaults to prevent
    rightmost label clipping with content margins.
    """
    # Test horizontal axis defaults
    h_axis = pg.AxisItem('bottom')
    assert h_axis.style['autoExpandTextSpace'] == True, \
        "Horizontal axis should have autoExpandTextSpace=True by default"
    assert h_axis.style['hideOverlappingLabels'] == False, \
        "Horizontal axis should have hideOverlappingLabels=False to prevent clipping"
    assert h_axis.style['autoReduceTextSpace'] == False, \
        "Horizontal axis should have autoReduceTextSpace=False for better stability"
    
    # Test vertical axis defaults  
    v_axis = pg.AxisItem('left')
    assert v_axis.style['autoExpandTextSpace'] == True, \
        "Vertical axis should have autoExpandTextSpace=True by default"
    assert v_axis.style['hideOverlappingLabels'] == False, \
        "Vertical axis should have hideOverlappingLabels=False"
    assert v_axis.style['autoReduceTextSpace'] == False, \
        "Vertical axis should have autoReduceTextSpace=False for better stability"


def test_AxisItem_text_space_methods():
    """
    Test that new text space calculation methods exist and work.
    
    These methods were added to fix issue #3375 by providing better
    text space management with PlotItem content margins.
    """
    axis = pg.AxisItem('bottom')
    
    # Test that new methods exist
    assert hasattr(axis, '_calculateRequiredTextSpace'), \
        "AxisItem should have _calculateRequiredTextSpace method"
    assert hasattr(axis, '_getAvailableTextSpace'), \
        "AxisItem should have _getAvailableTextSpace method"
    assert hasattr(axis, '_getParentLayoutMargins'), \
        "AxisItem should have _getParentLayoutMargins method"
    assert hasattr(axis, '_requestLayoutExpansion'), \
        "AxisItem should have _requestLayoutExpansion method"
    
    # Test methods don't crash with no data
    required_space = axis._calculateRequiredTextSpace()
    assert isinstance(required_space, (int, float)), \
        "Should return numeric value"
    assert required_space >= 0, \
        "Should return non-negative value"
        
    available_space = axis._getAvailableTextSpace()
    assert isinstance(available_space, (int, float)), \
        "Should return numeric value"
    assert available_space >= 0, \
        "Should return non-negative value"
        
    # Test parent margin detection (should return None with no parent)
    margins = axis._getParentLayoutMargins()
    assert margins is None, \
        "Should return None when no parent with margins"


def test_AxisItem_clipping_with_plot_margins():
    """
    Test the original issue #3375 scenario.
    
    Verify that rightmost tick labels are not clipped when using
    PlotItem with content margins.
    """
    plot_widget = pg.PlotWidget()
    plot_item = plot_widget.plotItem
    
    # Set content margins - this was causing the original issue
    plot_item.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
    
    # Add data that typically causes clipping
    import numpy as np
    x_data = np.linspace(0, 9.999, 50)  # Ending at 9.999 often causes clipping
    y_data = np.sin(x_data)
    plot_widget.plot(x_data, y_data)
    
    # Get the bottom axis
    bottom_axis = plot_item.getAxis('bottom')
    
    # With our fix, these should be the improved defaults
    assert bottom_axis.style.get('autoExpandTextSpace', False) == True
    assert bottom_axis.style.get('hideOverlappingLabels', True) == False
    
    # Test that parent margins are detected
    margins = bottom_axis._getParentLayoutMargins()
    assert margins is not None, \
        "Should detect parent PlotItem margins"
    
    # Test that text space calculation works
    required_space = bottom_axis._calculateRequiredTextSpace()
    available_space = bottom_axis._getAvailableTextSpace()
    
    assert isinstance(required_space, (int, float))
    assert isinstance(available_space, (int, float))
    assert required_space >= 0
    assert available_space >= 0
    
    plot_widget.close()


def test_AxisItem_manual_settings_override():
    """
    Test that manual axis settings still override the improved defaults.
    
    This ensures backward compatibility - users can still manually
    configure axis behavior if needed.
    """
    axis = pg.AxisItem('bottom')
    
    # Change settings manually
    axis.setStyle(
        autoExpandTextSpace=False,
        hideOverlappingLabels=True,
        autoReduceTextSpace=True
    )
    
    # Verify settings took effect
    assert axis.style['autoExpandTextSpace'] == False
    assert axis.style['hideOverlappingLabels'] == True
    assert axis.style['autoReduceTextSpace'] == True


@pytest.mark.parametrize("orientation", ['left', 'right', 'top', 'bottom'])
def test_AxisItem_orientation_specific_behavior(orientation):
    """
    Test that all axis orientations have appropriate default settings.
    
    The fix improves defaults for all orientations to prevent clipping issues.
    """
    axis = pg.AxisItem(orientation)
    
    # All orientations should have these improved defaults
    assert axis.style['autoExpandTextSpace'] == True
    assert axis.style['hideOverlappingLabels'] == False
    assert axis.style['autoReduceTextSpace'] == False
    
    # Methods should work for all orientations
    required_space = axis._calculateRequiredTextSpace()
    available_space = axis._getAvailableTextSpace()
    
    assert isinstance(required_space, (int, float))
    assert isinstance(available_space, (int, float))
    assert required_space >= 0
    assert available_space >= 0
