from typing import Any, Dict, Optional, Tuple, Union, TypedDict
import warnings

from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    ConfigKeyHint,
    ConfigValueHint,
    initItemStyle)
from ..Point import Point
from ..Qt import QtCore, QtWidgets
from ..graphicsItems.ViewBox.ViewBox import ViewBox
from .GraphicsObject import GraphicsObject
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .TextItem import TextItem

__all__ = ['ScaleBar']

Number = Union[float, int]
optsHint = TypedDict('optsHint',
                     {'faceColor'  : ConfigColorHint,
                      'labelColor' : ConfigColorHint,
                      'offset'     : Tuple[float, float],
                      'width'      : Number},
                     total=False)
# kwargs are not typed because mypy has not ye included Unpack[Typeddict]

class ScaleBar(GraphicsWidgetAnchor, GraphicsObject):
    """
    Displays a rectangular bar to indicate the relative scale of objects on the view.
    """
    def __init__(self, size: Number,
                       suffix: str='m',
                       **kwargs) -> None:
        """
        Displays a rectangular bar to indicate the relative scale of objects on the view.

        Parameters
        ----------
        size:
            Scale value to be displayed above the bar.
        suffix:
            Text to be added as unit of the scalebar
        **kwargs: optional
            style options , see setStyle() for accepted style parameters.
        """

        GraphicsObject.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        # Store style options in opts dict
        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'ScaleBar', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)

        # We keep old style parameters
        if 'brush' in kwargs:
            self._brush = kwargs['brush']
        else:
            self._brush = None
        if 'pen' in kwargs:
            self._pen = kwargs['pen']
        else:
            self._pen = None

        self.brush = fn.mkBrush(self._getBrush())
        self.pen = fn.mkPen(self._getPen())
        self.size = size

        self.bar = QtWidgets.QGraphicsRectItem()
        self.bar.setPen(self.pen)
        self.bar.setBrush(self.brush)
        self.bar.setParentItem(self)

        self.text = TextItem(text=fn.siFormat(size, suffix=suffix), anchor=(0.5,1))
        self.text.setParentItem(self)


    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################


    def getFaceColor(self) -> ConfigColorHint:
        return self.opts['faceColor']

    def setFaceColor(self, color: ConfigColorHint):
        self.opts['faceColor'] = color

    def getLabelColor(self) -> ConfigColorHint:
        return self.opts['labelColor']

    def setLabelColor(self, labelColor: ConfigColorHint):
        self.opts['labelColor'] = labelColor

    def setOffset(self, offset: Tuple[float, float]):
        """
        Set the offset position of the label relative to the bar.
        """
        self.opts['offset'] = offset

    def getOffset(self) -> Tuple[float, float]:
        """
        Get the offset position of the label relative to the bar.
        """
        return self.opts['offset']

    def setWidth(self, width: int) -> None:
        """
        Set the width of the scalebar.

        Parameters
        ----------
        width : int
            The width of the scalebar in px.

        Returns
        -------
        None
        """
        self.opts['width'] = width


    def getWidth(self) -> int:
        """
        Get the current width of the scalebar.

        Returns
        -------
        int
            The current width of the scalebar in px.
        """
        return self.opts['width']


    ## Since we keep two parameters for the same task, these 2 methods sort
    # that
    def _getBrush(self):
        if self._brush is None:
            return fn.mkBrush(self.opts['faceColor'])
        else:
            return self._brush

    def _getPen(self):
        if self._pen is None:
            return fn.mkPen(self.opts['labelColor'])
        else:
            return self._pen

    def setStyle(self, **kwargs) -> None:
        """
        Set the style of the ScaleBar.

        Parameters
        ----------
        faceColor:
            Color of the scalebar.
        labelColor:
            Color of the text label.
        offset (Tuple[float, float]):
            Text style.
        width (int):
            Width of the scalebar.
        """
        for k, v in kwargs.items():
            # If the key is a valid entry of the stylesheet
            if k in configStyle['ScaleBar'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))


    def changeParent(self) -> None:
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self.updateBar)
        self.updateBar()


    def updateBar(self) -> None:
        view = self.parentItem()
        if view is None:
            return
        p1 = view.mapFromViewToItem(self, QtCore.QPointF(0,0))
        p2 = view.mapFromViewToItem(self, QtCore.QPointF(self.size,0))
        w = (p2-p1).x()
        self.bar.setRect(QtCore.QRectF(-w, 0, w, self.opts['width']))
        self.text.setPos(-w/2., 0)


    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF()


    def setParentItem(self, p: ViewBox):
        ret = GraphicsObject.setParentItem(self, p)
        # if self.offset is not None:
        offset = Point(self.opts['offset'])
        anchorx = 1 if offset[0] <= 0 else 0
        anchory = 1 if offset[1] <= 0 else 0
        anchor = (anchorx, anchory)
        self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret
