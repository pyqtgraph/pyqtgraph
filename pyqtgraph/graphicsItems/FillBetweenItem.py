from ..Qt import QtGui
from .. import functions as fn

class FillBetweenItem(QtGui.QGraphicsPathItem):
    """
    GraphicsItem filling the space between two PlotDataItems.
    """
    def __init__(self, p1, p2, brush=None):
        QtGui.QGraphicsPathItem.__init__(self)
        self.p1 = p1
        self.p2 = p2
        p1.sigPlotChanged.connect(self.updatePath)
        p2.sigPlotChanged.connect(self.updatePath)
        if brush is not None:
            self.setBrush(fn.mkBrush(brush))
        self.setZValue(min(p1.zValue(), p2.zValue())-1)
        self.updatePath()

    def updatePath(self):
        p1 = self.p1.curve.path
        p2 = self.p2.curve.path
        path = QtGui.QPainterPath()
        path.addPolygon(p1.toSubpathPolygons()[0] + p2.toReversed().toSubpathPolygons()[0])
        self.setPath(path)
