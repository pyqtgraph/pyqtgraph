import warnings
from typing import Any, Optional, Union

from .. import functions as fn
from .. import configStyle
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor

__all__ = ['LabelItem']

class LabelItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.

    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """


    def __init__(self, text: str=' ',
                       parent: Optional[Any]=None,
                       angle: float=0,
                       **args: dict[str, Any]) -> None:
        GraphicsWidget.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.item = QtWidgets.QGraphicsTextItem(self)
        self.opts = {}
        self.opts.update(args)
        self._sizeHint = {}
        self.setText(text)
        self.setAngle(angle)

    def setAttr(self, attr: str,
                      value: Union[str, float, bool]) -> None:
        """
        Set a style property.
        See setText() for accepted style parameter.
        """
        self.opts[attr] = value

    def setText(self, text: str,
                      **args) -> None:
        """Set the text and text properties in the label.
        Accepts optional arguments for auto-generating a CSS style string.

        Args:
            text: Text to be displayed.
            args: Style arguments:
                color   (str): text color. Example: '#CCFF00'
                size    (float): text size in pt. Example: 8
                bold    (bool): deprecated. weight should be used instead, see below.
                weight  (str): Text weight. Must be: 'normal', 'bold', 'bolder', or 'lighter'
                italic  (bool): deprecated, style should be used instead, see below.
                style   (str): Text style. Must be: 'normal', 'italic', or 'oblique'
                justify (str): Text alignement. Must be: 'left', 'center', or 'right'
        """
        self.text = text
        opts = self.opts
        for k in args:
            opts[k] = args[k]

        optlist = []

        ## Color
        if 'color' not in opts:
            colorQt = fn.mkColor(configStyle['labelItem.color'])
        else:
            try:
                colorQt = fn.mkColor(opts['color'])
            except:
                raise ValueError('Given color argument:{} is not a proper color'.format(opts['color']))
        optlist.append('color: {}'.format(colorQt.name(QtGui.QColor.NameFormat.HexArgb)))

        ## Font-size
        # Currently we accept a float, a string or None
        # If None, we use the stylesheet labelItem.fontsize
        # If float, we interprete the given value as the font-size in pt
        # If sring, we interprete the given string as the css property value
        if 'size' not in self.opts:
            size = configStyle['labelItem.fontsize']
        else:
            try:
                size = float(opts['size'])
            except ValueError:
                warnings.warn('Argument "size" given as string is deprecated and should be set by a float instead.',
                              DeprecationWarning,
                              stacklevel=2)
                try:
                    size = float(opts['size'][:-2])
                except:
                    raise ValueError('Given size argument:{} is not a proper float or string'.format(opts['size']))
        optlist.append('font-size: {}pt'.format(size))

        ## Font-weight
        # Currently can be setted by two argument: "bold" deprecated and "weight"
        #       bold must be a boolean, True means "font-weight: bold" and False "font-weight: normal"
        # for weight we accept a string or None
        # If None, we use the stylesheet labelItem.fontweight
        # If sring, we interprete the given string as the css property value
        if 'bold' in self.opts:
            warnings.warn('Argument "bold" is deprecated, "weight" should be used instead.',
                          DeprecationWarning,
                          stacklevel=2)
            if isinstance(opts['bold'], bool):
                weight = {True:'bold', False:'normal'}[opts['bold']]
            else:
                raise ValueError('Given "bold" argument:{}, is not a boolean.'.format(opts['bold']))
        elif 'weight' in self.opts:
            if isinstance(weight, str):
                if weight not in ('normal, bold, bolder, lighter'):
                    raise ValueError('weight argument:{} must be "normal", "bold", "bolder", or "lighter"'.format(weight))
            else:
                raise ValueError('weight argument:{} is not a string'.format(opts['weight']))
            weight = opts['weight']
        else:
            weight = configStyle['labelItem.fontweight']
        optlist.append('font-weight: {}'.format(weight))

        ## Font-style
        # Currently can be setted by two argument: "italic" deprecated and "style"
        #       italic must be a boolean, True means "font-style: italic" and False "font-style: normal"
        # for style we accept a string or None
        # If None, we use the stylesheet labelItem.fontstyle
        # If sring, we interprete the given string as the css property value
        if 'italic' in self.opts:
            warnings.warn('Argument "italic" is deprecated, "style" should be used instead.',
                          DeprecationWarning,
                          stacklevel=2)
            if isinstance(opts['italic'], bool):
                style = {True:'italic', False:'normal'}[opts['italic']]
            else:
                raise ValueError('Given "italic" argument:{}, is not a boolean.'.format(opts['italic']))
        elif 'style' in self.opts:
            if isinstance(style, str):
                if style not in ('normal, bold, bolder, lighter'):
                    raise ValueError('style argument:{} must be "normal", "italic", or "oblique"'.format(style))
            else:
                raise ValueError('style argument:{} is not a string'.format(opts['style']))
            style = opts['style']
        else:
            style = configStyle['labelItem.fontstyle']
        optlist.append('font-style: {}'.format(style))

        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)

        self.item.setHtml(full)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()

    def resizeEvent(self, ev) -> None:
        #c1 = self.boundingRect().center()
        #c2 = self.item.mapToParent(self.item.boundingRect().center()) # + self.item.pos()
        #dif = c1 - c2
        #self.item.moveBy(dif.x(), dif.y())
        #print c1, c2, dif, self.item.pos()
        self.item.setPos(0,0)
        bounds = self.itemRect()
        left = self.mapFromItem(self.item, QtCore.QPointF(0,0)) - self.mapFromItem(self.item, QtCore.QPointF(1,0))
        rect = self.rect()


        ## Text-alignement
        # Currently can be setted by two argument: "justify" deprecated and "align"
        #       justify and align must be "left", "center", or "right"
        # for justify we accept a string or None
        # If None, we use the stylesheet labelItem.fontstyle
        # for align we accept a string or None
        # If None, we use the stylesheet labelItem.fontstyle
        if 'justify' in self.opts:
            warnings.warn('Argument "justify" is deprecated, "align" should be used instead.',
                          DeprecationWarning,
                          stacklevel=2)
            if isinstance(self.opts['justify'], str):
                if self.opts['justify'] in ('left', 'center', 'right'):
                    align = self.opts['justify']
            else:
                raise ValueError('Given "justify" argument:{} must be "left", "center", or "right".'.format(self.opts['justify']))
        elif 'align' in self.opts:
            if isinstance(self.opts['align'], str):
                if self.opts['align'] in ('left', 'center', 'right'):
                    align = self.opts['align']
            else:
                raise ValueError('Given "align" argument:{} must be "left", "center", or "right".'.format(self.opts['justify']))
        else:
            align = configStyle['labelItem.align']

        if align == 'left':
            if left.x() != 0:
                bounds.moveLeft(rect.left())
            if left.y() < 0:
                bounds.moveTop(rect.top())
            elif left.y() > 0:
                bounds.moveBottom(rect.bottom())

        elif align == 'center':
            bounds.moveCenter(rect.center())
            #bounds = self.itemRect()
            #self.item.setPos(self.width()/2. - bounds.width()/2., 0)
        elif align == 'right':
            if left.x() != 0:
                bounds.moveRight(rect.right())
            if left.y() < 0:
                bounds.moveBottom(rect.bottom())
            elif left.y() > 0:
                bounds.moveTop(rect.top())
            #bounds = self.itemRect()
            #self.item.setPos(self.width() - bounds.width(), 0)

        self.item.setPos(bounds.topLeft() - self.itemRect().topLeft())
        self.updateMin()

    def setAngle(self, angle: float) -> None:
        self.angle = angle
        self.item.resetTransform()
        self.item.setRotation(angle)
        self.updateMin()


    def updateMin(self) -> None:
        bounds = self.itemRect()
        self.setMinimumWidth(bounds.width())
        self.setMinimumHeight(bounds.height())

        self._sizeHint = {
            QtCore.Qt.SizeHint.MinimumSize: (bounds.width(), bounds.height()),
            QtCore.Qt.SizeHint.PreferredSize: (bounds.width(), bounds.height()),
            QtCore.Qt.SizeHint.MaximumSize: (-1, -1),  #bounds.width()*2, bounds.height()*2),
            QtCore.Qt.SizeHint.MinimumDescent: (0, 0)  ##?? what is this?
        }
        self.updateGeometry()

    def sizeHint(self, hint: int,
                       constraint: Any) -> QtCore.QSizeF:
        """
        Return the recommended size for the widget.

        Args:
            hint: Integer used to specify the minimum size of a graphics layout item:
                0 : MinimumSize
                1 : PreferredSize
                2 : MaximumSize
                3 : MinimumDescent
            constraint: no idea, not used
        """
        if hint not in self._sizeHint:
            return QtCore.QSizeF(0, 0)
        return QtCore.QSizeF(*self._sizeHint[hint])

    def itemRect(self) -> QtCore.QRectF :
        return self.item.mapRectToParent(self.item.boundingRect())

    #def paint(self, p, *args):
        #p.setPen(fn.mkPen('r'))
        #p.drawRect(self.rect())
        #p.setPen(fn.mkPen('g'))
        #p.drawRect(self.itemRect())

