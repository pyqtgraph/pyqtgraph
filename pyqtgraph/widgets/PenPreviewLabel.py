from pyqtgraph import QtWidgets, QtGui


class PenPreviewLabel(QtWidgets.QLabel):
    def __init__(self, param):
        super().__init__()
        self.param = param
        param.sigValueChanged.connect(self.update)

    @property
    def pen(self):
        return self.param.pen

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
        painter.end()
