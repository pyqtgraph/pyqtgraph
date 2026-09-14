from ..Qt import QtWidgets, QtGui
from ..functions import mkPen


class BrushPreviewLabel(QtWidgets.QLabel):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.brush = QtGui.QBrush(self.param.brush)
        param.sigValueChanging.connect(self.onBrushChanging)

    def onBrushChanging(self, param, val):
        self.brush = QtGui.QBrush(val)
        self.update()

    def paintEvent(self, ev):
        displaySize = self.size()
        w, h = displaySize.width(), displaySize.height()
        margin = 2

        painter = QtGui.QPainter(self)
        painter.setBrush(self.brush)
        painter.setPen(mkPen('k'))
        painter.drawRect(margin, margin, w - 2 * margin, h - 2 * margin)
        painter.end()
