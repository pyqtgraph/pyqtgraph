from .basetypes import WidgetParameterItem
from ..Parameter import Parameter
from ... import functions as fn
from ...Qt import QtWidgets, QtGui, QtCore
from ...widgets.PenSelectorDialog import PenSelectorDialog

class PenParameterItem(WidgetParameterItem):
    def __init__(self, param, depth):
        self.pdialog = PenSelectorDialog(fn.mkPen(param.pen))
        self.pdialog.setModal(True)
        self.pdialog.accepted.connect(self.penChangeFinished)
        super().__init__(param, depth)
        self.displayLabel.paintEvent = self.displayPaintEvent

    def makeWidget(self):
        self.button = QtWidgets.QPushButton()
        #larger button
        self.button.setFixedWidth(100)
        self.button.clicked.connect(self.buttonClicked)
        self.button.paintEvent = self.buttonPaintEvent
        self.button.value = self.value
        self.button.setValue = self.setValue
        self.button.sigChanged = None
        return self.button

    @property
    def pen(self):
        return self.pdialog.pen

    def value(self):
        return self.pen

    def setValue(self, pen):
        self.pdialog.updateParamFromPen(self.pdialog.param, pen)

    def updateDisplayLabel(self, value=None):
        super().updateDisplayLabel('')
        self.displayLabel.update()
        self.widget.update()

    def buttonClicked(self):
        #open up the pen selector dialog
        # Copy in case of rejection
        prePen = QtGui.QPen(self.pen)
        if self.pdialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self.pdialog.updateParamFromPen(self.pdialog.param, prePen)

    def penChangeFinished(self):
        self.param.setValue(self.pdialog.pen)

    def penPaintEvent(self, event, item):
        # draw item as usual
        type(item).paintEvent(item, event)

        path = QtGui.QPainterPath()
        displaySize = item.size()
        w, h = displaySize.width(), displaySize.height()
        # draw a squiggle with the pen
        path.moveTo(w * .2, h * .2)
        path.lineTo(w * .4, h * .8)
        path.cubicTo(w * .5, h * .1, w * .7, h * .1, w * .8, h * .8)

        painter = QtGui.QPainter(item)
        painter.setPen(self.pen)
        painter.drawPath(path)
        painter.end()

    def buttonPaintEvent(self, event):
        return self.penPaintEvent(event, self.button)

    def displayPaintEvent(self, event):
        return self.penPaintEvent(event, self.displayLabel)


class PenParameter(Parameter):
    """
    Controls the appearance of a QPen value.

    When `saveState` is called, the value is encoded as (color, width, style, capStyle, joinStyle, cosmetic)

    ============== ========================================================
    **Options:**
    color          pen color, can be any argument accepted by :func:`~pyqtgraph.mkColor` (defaults to black)
    width          integer width >= 0 (defaults to 1)
    style          String version of QPenStyle enum, i.e. 'SolidLine' (default), 'DashLine', etc.
    capStyle       String version of QPenCapStyle enum, i.e. 'SquareCap' (default), 'RoundCap', etc.
    joinStyle      String version of QPenJoinStyle enum, i.e. 'BevelJoin' (default), 'RoundJoin', etc.
    cosmetic       Boolean, whether or not the pen is cosmetic (defaults to True)
    ============== ========================================================
    """

    itemClass = PenParameterItem
    sigPenChanged = QtCore.Signal(object,object)

    def __init__(self, **opts):
        self.pen = fn.mkPen()
        self.penOptsParam = PenSelectorDialog.mkParam(self.pen)
        super().__init__(**opts)

    def saveState(self, filter=None):
        state = super().saveState(filter)
        overrideState = self.penOptsParam.saveState(filter)['children']
        state['value'] = tuple(s['value'] for s in overrideState.values())
        return state

    def _interpretValue(self, v):
        return self.mkPen(v)

    def setValue(self, value, blockSignal=None):
        if not fn.eq(value, self.pen):
            value = self.mkPen(value)
            PenSelectorDialog.updateParamFromPen(self.penOptsParam, value)
        return super().setValue(self.pen, blockSignal)

    def applyOptsToPen(self, **opts):
        # Transform opts into a value for the current pen
        paramNames = set(opts).intersection(self.penOptsParam.names)
        # Value should be overridden by opts
        with self.treeChangeBlocker():
            if 'value' in opts:
                pen = self.mkPen(opts.pop('value'))
                if not fn.eq(pen, self.pen):
                    PenSelectorDialog.updateParamFromPen(self.penOptsParam, pen)
            penOpts = {}
            for kk in paramNames:
                penOpts[kk] = opts[kk]
                self.penOptsParam[kk] = opts[kk]
        return penOpts

    def setOpts(self, **opts):
        # Transform opts into a value
        penOpts = self.applyOptsToPen(**opts)
        if penOpts:
            self.setValue(self.pen)
        return super().setOpts(**opts)

    def mkPen(self, *args, **kwargs):
        """Thin wrapper around fn.mkPen which accepts the serialized state from saveState"""
        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == len(self.penOptsParam.childs):
            opts = dict(zip(self.penOptsParam.names, args[0]))
            self.applyOptsToPen(**opts)
            args = (self.pen,)
            kwargs = {}
        return fn.mkPen(*args, **kwargs)
