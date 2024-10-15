from typing import Union

from .. import functions as fn
from ..Qt import QtGui, QtWidgets, QtCore
from .PlotCurveItem import PlotCurveItem
from .PlotDataItem import PlotDataItem

__all__ = ['FillBetweenItem']

class FillBetweenItem(QtWidgets.QGraphicsPathItem):
    """
    GraphicsItem filling the space between two PlotDataItems.
    """
    def __init__(
        self,
        curve1: Union[PlotDataItem, PlotCurveItem],
        curve2: Union[PlotDataItem, PlotCurveItem],
        brush=None,
        pen=None,
        fillRule: QtCore.Qt.FillRule=QtCore.Qt.FillRule.OddEvenFill
    ):
        """FillBetweenItem fills a region between two curves with a specified
        :class:`~QtGui.QBrush`. 

        Parameters
        ----------
        curve1 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill from
        curve2 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill to
        brush : color_like, optional
            Arguments accepted by :func:`~pyqtgraph.mkBrush`, used
            to create the :class:`~QtGui.QBrush` instance used to draw the item
            by default None
        pen : color_like, optional
            Arguments accepted by :func:`~pyqtgraph.mkColor`, used
            to create the :class:`~QtGui.QPen` instance used to draw the item
            by default ``None``
        fillRule : QtCore.Qt.FillRule, optional
            FillRule to be applied to the underlying :class:`~QtGui.QPainterPath`
            instance, by default ``QtCore.Qt.FillRule.OddEvenFill``

        Raises
        ------
        ValueError
            Raised when ``None`` is passed in as either ``curve1``
            or ``curve2``
        TypeError
            Raised when either ``curve1`` or ``curve2`` is not either
            :class:`~pyqtgraph.PlotDataItem` or :class:`~pyqtgraph.PlotCurveItem`
        """
        super().__init__()
        self.curves = None
        self._fillRule = fillRule
        if curve1 is not None and curve2 is not None:
            self.setCurves(curve1, curve2)
        elif curve1 is not None or curve2 is not None:
            raise ValueError("Must specify two curves to fill between.")

        if brush is not None:
            self.setBrush(brush)
        self.setPen(pen)
        self.updatePath()

    def fillRule(self):
        return self._fillRule

    def setFillRule(self, fillRule: QtCore.Qt.FillRule=QtCore.Qt.FillRule.OddEvenFill):
        """Set the underlying :class:`~QtGui.QPainterPath` to the specified 
        :class:`~QtCore.Qt.FillRule`

        This can be useful for allowing in the filling of voids.

        Parameters
        ----------
        fillRule : QtCore.Qt.FillRule
            A member of the :class:`~QtCore.Qt.FillRule` enum
        """
        self._fillRule = fillRule
        self.updatePath()
        
    def setBrush(self, *args, **kwds):
        """Change the fill brush. Accepts the same arguments as :func:`~pyqtgraph.mkBrush`
        """
        QtWidgets.QGraphicsPathItem.setBrush(self, fn.mkBrush(*args, **kwds))
        
    def setPen(self, *args, **kwds):
        """Change the fill pen. Accepts the same arguments as :func:`~pyqtgraph.mkColor`
        """
        QtWidgets.QGraphicsPathItem.setPen(self, fn.mkPen(*args, **kwds))

    def setCurves(
        self,
        curve1: Union[PlotDataItem, PlotCurveItem],
        curve2: Union[PlotDataItem, PlotCurveItem]
    ):
        """Method to set the Curves to draw the FillBetweenItem between

        Parameters
        ----------
        curve1 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill from
        curve2 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill to
    
        Raises
        ------
        TypeError
            Raised when input arguments are not either :class:`~pyqtgraph.PlotDataItem` or
            :class:`~pyqtgraph.PlotCurveItem`
        """        
        if self.curves is not None:
            for c in self.curves:
                try:
                    c.sigPlotChanged.disconnect(self.curveChanged)
                except (TypeError, RuntimeError):
                    pass

        curves = [curve1, curve2]
        for c in curves:
            if not isinstance(c, (PlotDataItem, PlotCurveItem)):
                raise TypeError("Curves must be PlotDataItem or PlotCurveItem.")
        self.curves = curves
        curve1.sigPlotChanged.connect(self.curveChanged)
        curve2.sigPlotChanged.connect(self.curveChanged)
        self.setZValue(min(curve1.zValue(), curve2.zValue())-1)
        self.curveChanged()

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
        path.setFillRule(self.fillRule())   

        ps1 = paths[0].toSubpathPolygons()
        ps2 = paths[1].toReversed().toSubpathPolygons()
        ps2.reverse()

        if len(ps1) == 0 or len(ps2) == 0:
            self.setPath(QtGui.QPainterPath())
            return

        for p1, p2 in zip(ps1, ps2):
            path.addPolygon(p1 + p2)

        self.setPath(path)
