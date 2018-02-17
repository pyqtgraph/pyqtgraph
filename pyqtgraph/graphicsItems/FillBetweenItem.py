from ..Qt import QtGui
from .. import functions as fn
from .PlotDataItem import PlotDataItem
from .PlotCurveItem import PlotCurveItem

class FillBetweenItem(QtGui.QGraphicsPathItem):
    """
    GraphicsItem filling the space between two PlotDataItems.
    """
    def __init__(self, curve1=None, curve2=None, brush=None, pen=None):
        QtGui.QGraphicsPathItem.__init__(self)
        self.curves = None
        if curve1 is not None and curve2 is not None:
            self.setCurves(curve1, curve2)
        elif curve1 is not None or curve2 is not None:
            raise Exception("Must specify two curves to fill between.")

        if brush is not None:
            self.setBrush(brush)
        self.setPen(pen)
        self.updatePath()
        
    def setBrush(self, *args, **kwds):
        QtGui.QGraphicsPathItem.setBrush(self, fn.mkBrush(*args, **kwds))
        
    def setPen(self, *args, **kwds):
        QtGui.QGraphicsPathItem.setPen(self, fn.mkPen(*args, **kwds))

    def setCurves(self, curve1, curve2):
        """Set the curves to fill between.
        
        Arguments must be instances of PlotDataItem or PlotCurveItem.
        
        Added in version 0.9.9
        """
        if self.curves is not None:
            for c in self.curves:
                try:
                    c.sigPlotChanged.disconnect(self.curveChanged)
                except (TypeError, RuntimeError):
                    pass

        curves = [curve1, curve2]
        for c in curves:
            if not isinstance(c, PlotDataItem) and not isinstance(c, PlotCurveItem):
                raise TypeError("Curves must be PlotDataItem or PlotCurveItem.")
        self.curves = curves
        curve1.sigPlotChanged.connect(self.curveChanged)
        curve2.sigPlotChanged.connect(self.curveChanged)
        self.setZValue(min(curve1.zValue(), curve2.zValue())-1)
        self.curveChanged()

    def setBrush(self, *args, **kwds):
        """Change the fill brush. Acceps the same arguments as pg.mkBrush()"""
        QtGui.QGraphicsPathItem.setBrush(self, fn.mkBrush(*args, **kwds))

    def curveChanged(self):
        self.updatePath()

    def updatePath(self):
        if self.curves is None:
            self.setPath(QtGui.QPainterPath())
            return
        paths = []
        for c in self.curves:
            if isinstance(c, PlotDataItem):
                paths.append(c.curve.getPath())
            elif isinstance(c, PlotCurveItem):
                paths.append(c.getPath())

        path = QtGui.QPainterPath()
        transform = QtGui.QTransform()
        ps1 = paths[0].toSubpathPolygons(transform)
        ps2 = paths[1].toReversed().toSubpathPolygons(transform)
        ps2.reverse()
        if len(ps1) == 0 or len(ps2) == 0:
            self.setPath(QtGui.QPainterPath())
            return
        
            
        for p1, p2 in zip(ps1, ps2):
            path.addPolygon(p1 + p2)
        self.setPath(path)
