from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkPen


class PenPreviewLabel(QtWidgets.QLabel):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.pen = QtGui.QPen(self.param.pen)
        param.sigValueChanging.connect(self.onPenChanging)

    def onPenChanging(self, param, val):
        self.pen = QtGui.QPen(val)
        self.update()

    def paintEvent(self, ev):
        path = QtGui.QPainterPath()
        displaySize = self.size()
        w, h = displaySize.width(), displaySize.height()
        # draw a squiggle with the pen
        path.moveTo(w * .2, h * .2)
        path.lineTo(w * .4, h * .8)
        path.cubicTo(w * .5, h * .1, w * .7, h * .1, w * .8, h * .8)

        painter = QtGui.QPainter(self)
        painter.setPen(self.pen)
        painter.drawPath(path)

        # No indication of "cosmetic" from just the paint path, so add something extra in that case
        if self.pen.isCosmetic():
            painter.setPen(mkPen('k'))
            painter.drawText(QtCore.QPointF(w * 0.81, 12), 'C')
        painter.end()
