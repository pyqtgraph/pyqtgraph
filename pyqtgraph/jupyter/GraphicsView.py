import jupyter_rfb
import numpy as np

from .. import functions as fn
from .. import graphicsItems, widgets
from ..Qt import QtCore, QtGui

__all__ = ['GraphicsLayoutWidget', 'PlotWidget']

KMLUT = {
    x : getattr(QtCore.Qt.KeyboardModifier, x + "Modifier")
    for x in ["Shift", "Control", "Alt", "Meta"]
}

MBLUT = {
    k : getattr(QtCore.Qt.MouseButton, v + "Button")
    for (k, v) in zip(
        range(6),
        ["No", "Left", "Right", "Middle", "Back", "Forward"]
    )
}

TYPLUT = {
    "pointer_down" : QtCore.QEvent.Type.MouseButtonPress,
    "pointer_up" : QtCore.QEvent.Type.MouseButtonRelease,
    "pointer_move" : QtCore.QEvent.Type.MouseMove,
}

def get_buttons(evt_buttons):
    NoButton = QtCore.Qt.MouseButton.NoButton
    btns = NoButton
    for x in evt_buttons:
        btns |= MBLUT.get(x, NoButton)
    return btns

def get_modifiers(evt_modifiers):
    NoModifier = QtCore.Qt.KeyboardModifier.NoModifier
    mods = NoModifier
    for x in evt_modifiers:
        mods |= KMLUT.get(x, NoModifier)
    return mods


class GraphicsView(jupyter_rfb.RemoteFrameBuffer):
    """jupyter_rfb.RemoteFrameBuffer sub-class that wraps around
    :class:`GraphicsView <pyqtgraph.GraphicsView>`.

    Generally speaking, there is no Qt event loop running. The implementation works by
    requesting a render() of the scene. Thus things that would work for exporting
    purposes would be expected to work here. Things that are not part of the scene
    would not work, e.g. context menus, tooltips.

    This class should not be used directly. Its corresponding sub-classes
    :class:`GraphicsLayoutWidget <pyqtgraph.jupyter.GraphicsLayoutWidget>` and
    :class:`PlotWidget <pyqtgraph.jupyter.PlotWidget>` should be used instead."""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.gfxView = widgets.GraphicsView.GraphicsView()
        self.logical_size = int(self.css_width[:-2]), int(self.css_height[:-2])
        self.pixel_ratio = 1.0
        # self.gfxView.resize(*self.logical_size)
        # self.gfxView.show()
        # self.gfxView.resizeEvent(None)

    def get_frame(self):
        w, h = self.logical_size
        dpr = self.pixel_ratio
        buf = np.empty((int(h * dpr), int(w * dpr), 4), dtype=np.uint8)
        qimg = fn.ndarray_to_qimage(buf, QtGui.QImage.Format.Format_RGBX8888)
        qimg.fill(QtCore.Qt.GlobalColor.transparent)
        qimg.setDevicePixelRatio(dpr)
        painter = QtGui.QPainter(qimg)
        self.gfxView.render(painter, self.gfxView.viewRect(), self.gfxView.rect())
        painter.end()
        return buf
    
    def handle_event(self, event):
        event_type = event["event_type"]

        if event_type == "resize":
            oldSize = QtCore.QSize(*self.logical_size)
            self.logical_size = int(event["width"]), int(event["height"])
            self.pixel_ratio = event["pixel_ratio"]
            self.gfxView.resize(*self.logical_size)
            newSize = QtCore.QSize(*self.logical_size)
            self.gfxView.resizeEvent(QtGui.QResizeEvent(newSize, oldSize))
        elif event_type in ["pointer_down", "pointer_up", "pointer_move"]:
            btn = MBLUT.get(event["button"], None)
            if btn is None:    # ignore unsupported buttons
                return
            pos = QtCore.QPointF(event["x"], event["y"])
            btns = get_buttons(event["buttons"])
            mods = get_modifiers(event["modifiers"])
            typ = TYPLUT[event_type]
            evt = QtGui.QMouseEvent(typ, pos, pos, btn, btns, mods)
            QtCore.QCoreApplication.sendEvent(self.gfxView.viewport(), evt)
            self.request_draw()
        elif event_type == "wheel":
            pos = QtCore.QPointF(event["x"], event["y"])
            pixdel = QtCore.QPoint()
            scale = -1.0    # map JavaScript wheel to Qt wheel
            angdel = QtCore.QPoint(int(event["dx"] * scale), int(event["dy"] * scale))
            btns = get_buttons([])
            mods = get_modifiers(event["modifiers"])
            phase = QtCore.Qt.ScrollPhase.NoScrollPhase
            inverted = False
            evt = QtGui.QWheelEvent(pos, pos, pixdel, angdel, btns, mods, phase, inverted)
            QtCore.QCoreApplication.sendEvent(self.gfxView.viewport(), evt)


def connect_viewbox_redraw(vb, request_draw):
    # connecting these signals is enough to support zoom/pan
    # but not enough to support the various graphicsItems
    # that react to mouse events

    vb.sigRangeChanged.connect(request_draw)
    # zoom / pan
    vb.sigRangeChangedManually.connect(request_draw)
    # needed for "auto" button
    vb.sigStateChanged.connect(request_draw)
    # note that all cases of sig{X,Y}RangeChanged being emitted
    # are also followed by sigRangeChanged or sigStateChanged
    vb.sigTransformChanged.connect(request_draw)


class GraphicsLayoutWidget(GraphicsView):
    """jupyter_rfb analogue of
    :class:`GraphicsLayoutWidget <pyqtgraph.GraphicsLayoutWidget>`."""

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.gfxLayout = graphicsItems.GraphicsLayout.GraphicsLayout()
        for n in [
            'nextRow', 'nextCol', 'nextColumn', 'addItem', 'getItem',
            'addLayout', 'addLabel', 'removeItem', 'itemIndex', 'clear'
        ]:
            setattr(self, n, getattr(self.gfxLayout, n))
        self.gfxView.setCentralItem(self.gfxLayout)

    def addPlot(self, *args, **kwds):
        kwds["enableMenu"] = False
        plotItem = self.gfxLayout.addPlot(*args, **kwds)
        connect_viewbox_redraw(plotItem.getViewBox(), self.request_draw)
        return plotItem

    def addViewBox(self, *args, **kwds):
        kwds["enableMenu"] = False
        vb = self.gfxLayout.addViewBox(*args, **kwds)
        connect_viewbox_redraw(vb, self.request_draw)
        return vb


class PlotWidget(GraphicsView):
    """jupyter_rfb analogue of
    :class:`PlotWidget <pyqtgraph.PlotWidget>`."""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        plotItem = graphicsItems.PlotItem.PlotItem(enableMenu=False)
        self.gfxView.setCentralItem(plotItem)
        connect_viewbox_redraw(plotItem.getViewBox(), self.request_draw)
        self.plotItem = plotItem

    def getPlotItem(self):
        return self.plotItem

    def __getattr__(self, attr):
        # kernel crashes if we don't skip attributes starting with '_'
        if attr.startswith('_'):
            return super().__getattr__(attr)

        # implicitly wrap methods from plotItem
        if hasattr(self.plotItem, attr):
            m = getattr(self.plotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise AttributeError(attr)
