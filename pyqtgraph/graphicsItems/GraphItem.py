import numpy as np
from typing import Tuple, TypedDict

from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    initItemStyle)
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import ScatterPlotItem

__all__ = ['GraphItem']


optsHint = TypedDict('optsHint',
                     {'background' : ConfigColorHint,
                      'lineColor'  : ConfigColorHint,
                      'antialias'  : bool},
                     total=False)
# kwargs are not typed because mypy has not ye included Unpack[Typeddict]


class GraphItem(GraphicsObject):
    """A GraphItem displays graph information as
    a set of nodes connected by lines (as in 'graph theory', not 'graphics').
    Useful for drawing networks, trees, etc.
    """

    def __init__(self, **kwargs) -> None:
        GraphicsObject.__init__(self)
        self.scatter = ScatterPlotItem()
        self.scatter.setParentItem(self)
        self.adjacency = None
        self.pos = None
        self.picture = None
        self.pen = 'default'
        self.setData(**kwargs)

        # Store style options in opts dict
        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'GraphItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)


    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    def setBackground(self, background: ConfigColorHint) -> None:
        """
        Set the background.
        """
        self.opts['background'] = background

    def background(self) -> ConfigColorHint:
        """
        Get the current background.
        """
        return self.opts['background']

    def setLineColor(self, lineColor: ConfigColorHint) -> None:
        """
        Set the lineColor.
        """
        self.opts['lineColor'] = lineColor

    def linecolor(self) -> ConfigColorHint:
        """
        Get the current lineColor.
        """
        return self.opts['lineColor']

    def setAntialias(self, antialias: bool) -> None:
        """
        Set the antialiasing
        """
        self.opts['antialias'] = antialias

    def antialias(self) -> bool:
        """
        Get if antialiasing
        """
        return self.opts['antialias']

    def setPen(self, *args, **kwargs) -> None:
        """
        Set the pen used to draw graph lines.
        May be:

          * None to disable line drawing
          * Record array with fields (red, green, blue, alpha, width)
          * Any set of arguments and keyword arguments accepted by
            :func:`mkPen <pyqtgraph.mkPen>`.
          * 'default' to use the default color.
        """
        if len(args) == 1 and len(kwargs) == 0:
            self.pen = args[0]
        else:
            self.pen = fn.mkPen(*args, **kwargs)
        self.picture = None
        self.update()

    def setStyle(self, **kwargs) -> None:
        """
        Set the style of the GraphItem.

        Parameters
        ----------
        background: ConfigColorHint or None, optional
            Color of the background.
            Any single argument accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        lineColor : ConfigColorHint or None, optional
            color of the lines between connected nodes.
            Any single argument accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        antialias : bool or None, optional
            If True, use antialiasing.

        Notes
        -----
        The parameters that are not provided will not be modified.

        Examples
        --------
        >>> setStyle(lineColor='w', background='k')
        """
        for k, v in kwargs.items():
            # If the key is a valid entry of the stylesheet
            if k in configStyle['LegendItem'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))

    ##############################################################
    #
    #                   Item
    #
    ##############################################################

    def setData(self, **kwargs) -> None:
        """
        Change the data displayed by the graph.

        ==============  =======================================================================
        **Arguments:**
        pos             (N,2) array of the positions of each node in the graph.
        adj             (M,2) array of connection data. Each row contains indexes
                        of two nodes that are connected or None to hide lines
        pen             The pen to use when drawing lines between connected
                        nodes. May be one of:

                          * QPen
                          * a single argument to pass to pg.mkPen
                          * a record array of length M
                            with fields (red, green, blue, alpha, width). Note
                            that using this option may have a significant performance
                            cost.
                          * None (to disable connection drawing)
                          * 'default' to use the default foreground color.

        symbolPen       The pen(s) used for drawing nodes.
        symbolBrush     The brush(es) used for drawing nodes.
        ``**opts``      All other keyword arguments are given to
                        :func:`ScatterPlotItem.setData() <pyqtgraph.ScatterPlotItem.setData>`
                        to affect the appearance of nodes (symbol, size, brush,
                        etc.)
        ==============  =======================================================================
        """
        if 'adj' in kwargs:
            self.adjacency = kwargs.pop('adj')
            if hasattr(self.adjacency, '__len__') and len(self.adjacency) == 0:
                self.adjacency = None
            elif self.adjacency is not None and self.adjacency.dtype.kind not in 'iu':
                raise Exception("adjacency must be None or an array of either int or unsigned type.")
            self._update()
        if 'pos' in kwargs:
            self.pos = kwargs['pos']
            self._update()
        if 'pen' in kwargs:
            self.setPen(kwargs.pop('pen'))
            self._update()

        if 'symbolPen' in kwargs:
            kwargs['pen'] = kwargs.pop('symbolPen')
        if 'symbolBrush' in kwargs:
            kwargs['brush'] = kwargs.pop('symbolBrush')
        self.scatter.setData(**kwargs)
        self.informViewBoundsChanged()

    def _update(self) -> None:
        self.picture = None
        self.prepareGeometryChange()
        self.update()

    def generatePicture(self) -> None:
        self.picture = QtGui.QPicture()
        if self.pen is None or self.pos is None or self.adjacency is None:
            return

        p = QtGui.QPainter(self.picture)
        try:
            pts = self.pos[self.adjacency]
            pen = self.pen
            if isinstance(pen, np.ndarray):
                lastPen = None
                for i in range(pts.shape[0]):
                    pen = self.pen[i]
                    if lastPen is None or np.any(pen != lastPen):
                        lastPen = pen
                        if pen.dtype.fields is None:
                            p.setPen(fn.mkPen(color=(pen[0], pen[1], pen[2], pen[3]), width=1))
                        else:
                            p.setPen(fn.mkPen(color=(pen['red'], pen['green'], pen['blue'], pen['alpha']), width=pen['width']))
                    p.drawLine(QtCore.QPointF(*pts[i][0]), QtCore.QPointF(*pts[i][1]))
            else:
                if pen == 'default':
                    pen = configStyle['GraphItem']['lineColor']
                p.setPen(fn.mkPen(pen))
                pts = pts.reshape((pts.shape[0]*pts.shape[1], pts.shape[2]))
                path = fn.arrayToQPath(x=pts[:,0], y=pts[:,1], connect='pairs')
                p.drawPath(path)
        finally:
            p.end()

    def paint(self, p: QtGui.QPainter, *args) -> None:
        if self.picture is None:
            self.generatePicture()
        if configStyle['GraphItem']['antialias']:
            p.setRenderHint(p.RenderHint.Antialiasing)
        self.picture.play(p)

    def boundingRect(self) -> QtCore.QRectF:
        return self.scatter.boundingRect()

    def dataBounds(self, *args, **kwargs) -> Tuple[float, float]:
        return self.scatter.dataBounds(*args, **kwargs)

    def pixelPadding(self) -> float:
        return self.scatter.pixelPadding()
