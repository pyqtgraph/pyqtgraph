from typing import Any, Dict, Optional, Tuple, TypedDict, Union
import warnings

from .. import functions as fn
from .. import configStyle
from ..style.core import configHint, configColorHint
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor

__all__ = ['LabelItem']


optsHint = TypedDict('optsHint',
                     {'color' : configColorHint,
                      'fontsize' : float,
                      'fontweight' : str,
                      'fontstyle' : str,
                      'align' : str},
                     total=False)


class LabelItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.

    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """


    def __init__(self, text: str=' ',
                       parent: Optional[Any]=None,
                       angle: float=0,
                       **kwargs: configHint) -> None:
        """
        Item to display text.

        Parameters
        ----------
        text: optional
            Text to be displayed, by default ' '
        parent: optional
            Parent item, by default None
        angle: optional
            angle of the displayed text in degrees, by default 0
        *args: optional
            style options , see setStyles() for accepted style parameters.
        """

        GraphicsWidget.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.item = QtWidgets.QGraphicsTextItem(self)

        # Store style options in opts dict
        self.opts: optsHint = {}
        # Get default stylesheet
        self._initStyle()
        # Update style
        self.setStyles(*kwargs)

        self._sizeHint: Dict[int, Tuple[float, float]] = {}
        self.setText(text)
        self.setAngle(angle)


    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################


    ## Font size
    def setFontsize(self, fontsize: float) -> None:

        if not isinstance(fontsize, float):
            raise ValueError('fontsize argument:{} is not a float'.format(fontsize))

        self.opts['fontsize'] = fontsize


    def getFontsize(self) -> float:
        return self.opts['fontsize']


    ## Font weight
    def setFontweight(self, fontweight: str) -> None:

        if isinstance(fontweight, str):
            if fontweight not in ('normal, bold, bolder, lighter'):
                raise ValueError('fontweight argument:{} must be "normal", "bold", "bolder", or "lighter"'.format(fontweight))
        else:
            raise ValueError('fontweight argument:{} is not a string'.format(fontweight))

        self.opts['fontweight'] = fontweight


    def getFontweight(self) -> str:
        return self.opts['fontweight']

    ## Font style
    def setFontstyle(self, fontstyle: str) -> None:

        if isinstance(fontstyle, str):
            if fontstyle not in ('normal, bold, bolder, lighter'):
                raise ValueError('fontstyle argument:{} must be "normal", "italic", or "oblique"'.format(fontstyle))
        else:
            raise ValueError('style argument:{} is not a string'.format(fontstyle))

        self.opts['fontweight'] = fontstyle


    def getFontstyle(self) -> str:
        return self.opts['fontstyle']

    ## Alignement
    def setAlignement(self, align: str) -> None:

        if isinstance(self.opts['align'], str):
            if align in ('left', 'center', 'right'):
                raise ValueError('Given "align" argument:{} must be "left", "center", or "right".'.format(align))
        else:
            raise ValueError('align argument:{} is not a string'.format(align))

        self.opts['align'] = align


    def getAlignement(self) -> str:
        return self.opts['align']

    ## Color
    def setColor(self, color: configColorHint) -> None:
        self.opts['color'] = color


    def getColor(self) -> configColorHint:
        return self.opts['color']


    def _initStyle(self) -> None:
        """
        Add to internal opts dict all labelItem style options from the stylesheet.
        """

        # Since we are using a typedDict, I couldn't find another way that
        # to assign the keys manually...
        # If there is a better way, I am in
        for key, val in configStyle.items():
            if key=='labelItem.color':
                self.opts['color'] = val
            elif key=='labelItem.fontsize':
                self.opts['fontsize'] = val
            elif key=='labelItem.fontweight':
                self.opts['fontweight'] = val
            elif key=='labelItem.fontstyle':
                self.opts['fontstyle'] = val
            elif key=='labelItem.align':
                self.opts['align'] = val


    def setStyle(self, attr: str,
                       value: Union[str, float, bool]) -> None:
        """
        Set a single style property.
        See:
            - setStyles() for all accepted style parameter.
            - stylesheet.

        Parameters
        ----------
        attr:
            style parameter to change
        value:
            its new value
        """

        # If the attr is a valid entry of the stylesheet
        if attr in (key[10:] for key in configStyle.keys() if key[:9]=='labelItem'):
            fun = getattr(self, 'set{}{}'.format(attr[:1].upper(), attr[1:]))
            fun(value)
        # For backward compatibility, we also accept some old attr values
        elif attr=='size':
            if isinstance(value, str):
                self._setSize(value)
            else:
                raise ValueError('Given size argument:{} is not a string.'.format(value))
        elif attr=='bold':
            if isinstance(value, bool):
                self._setBold(value)
            else:
                raise ValueError('Given "bold" argument:{}, is not a boolean.'.format(value))
        elif attr=='italic':
            if isinstance(value, bool):
                self._setItalic(value)
            else:
                raise ValueError('Given "italic" argument:{}, is not a boolean.'.format(value))
        elif attr=='justify':
            if isinstance(value, str):
                self._setJustify(value)
            else:
                raise ValueError('Given "italic" argument:{}, is not a boolean.'.format(value))
        else:
            raise ValueError('Your "attr" argument: "{}" is not recognized'.format(value))


    def setStyles(self, **kwargs):
        """
        Set the style of the LabelItem.

        Parameters
        ----------
        color (str):
            Text color. Example: '#CCFF00'.
        fontsize (float):
            Text size in pt.
        fontweight {'normal', 'bold', 'bolder' 'lighter'}:
            Text weight.
        fontstyle {'normal', 'italic' 'oblique'}:
            Text style.
        align {'left', 'center', 'right'}:
            Text alignement.

        Deprecated parameters
        size (float):
            text size in pt. Example: 8.
            (Deprecated:
            size (str) '8pt' allowed for compatibility).
        bold (bool):
            deprecated.
            weight should be used instead, see below.
        italic (bool):
            deprecated, style should be used instead, see below.
        justify (str):
            Text alignement.
            Must be: 'left', 'center', or 'right'
        """

        for k, v in kwargs.items():
            self.setStyle(k, v)


    ##############################################################
    #
    #                   Deprecated style methods
    #
    ##############################################################

    ## bold
    def _setBold(self, bold: bool) -> None:

        warnings.warn('Argument "bold" is deprecated, "fontweight" should be used instead.',
                       DeprecationWarning,
                       stacklevel=2)

        self.opts['fontweight'] = {True:'bold', False:'normal'}[bold]


    ## size
    def _setSize(self, size: str) -> None:

        warnings.warn('Argument "size" given as string is deprecated, "fontsize" should be used instead.',
                      DeprecationWarning,
                      stacklevel=2)
        try:
            sizeFloat = float(size[:-2])
        except:
            raise ValueError('Given size argument:{} is not a proper size in pt.'.format(size[:-2]))

        self.opts['fontsize'] = sizeFloat


    ## italic
    def _setItalic(self, italic: bool) -> None:

        warnings.warn('Argument "italic" is deprecated, "fontstyle" should be used instead.',
                        DeprecationWarning,
                        stacklevel=2)

        self.opts['fontstyle'] = {True:'italic', False:'normal'}[italic]


    ## justify
    def _setJustify(self, justify: str) -> None:

        warnings.warn('Argument "justify" is deprecated, "align" should be used instead.',
                        DeprecationWarning,
                        stacklevel=2)

        if justify in ('left', 'center', 'right'):
            align = justify
        else:
            raise ValueError('Given "justify" argument:{} must be "left", "center", or "right".'.format(justify))

        self.opts['align'] = align


    def setAttr(self, attr: str,
                      value: Union[str, float, bool]) -> None:
        """
        Deprecated, please use "setStyle".
        Set a style property.
        """

        warnings.warn('Method "setAttr" is deprecated. Use "setStyle" instead',
                        DeprecationWarning,
                        stacklevel=2)
        self.setStyle(attr, value)


    ##############################################################
    #
    #                   Text and angle
    #
    ##############################################################


    def setText(self, text: str,
                      **kwargs) -> None:
        """
        Set the text and text properties in the label.
        """

        self.text = text
        self.setStyles(**kwargs)

        optlist = []
        optlist.append('color: {}'.format(fn.mkColor(self.opts['color']).name(QtGui.QColor.NameFormat.HexArgb)))
        optlist.append('font-size: {}pt'.format(self.opts['fontsize']))
        optlist.append('font-weight: {}'.format(self.opts['fontweight']))
        optlist.append('font-style: {}'.format(self.opts['fontstyle']))
        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)
        self.item.setHtml(full)

        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()


    def resizeEvent(self, ev: Optional[QtWidgets.QGraphicsSceneResizeEvent]) -> None:
        """
        Set the position of the LabelItem considering alignement specified either
            in the item via "justify" (deprecated) or "align"
            in the stylesheet via "labelItem.align"

        Once the position is reset, call updateMin().
        """

        self.item.setPos(0,0)
        bounds = self.itemRect()
        left = self.mapFromItem(self.item, QtCore.QPointF(0,0)) - self.mapFromItem(self.item, QtCore.QPointF(1,0))
        rect = self.rect()

        if self.opts['align'] == 'left':
            if left.x() != 0:
                bounds.moveLeft(rect.left())
            if left.y() < 0:
                bounds.moveTop(rect.top())
            elif left.y() > 0:
                bounds.moveBottom(rect.bottom())
        elif self.opts['align'] == 'center':
            bounds.moveCenter(rect.center())
        elif self.opts['align'] == 'right':
            if left.x() != 0:
                bounds.moveRight(rect.right())
            if left.y() < 0:
                bounds.moveBottom(rect.bottom())
            elif left.y() > 0:
                bounds.moveTop(rect.top())

        self.item.setPos(bounds.topLeft() - self.itemRect().topLeft())
        self.updateMin()


    def setAngle(self, angle: float) -> None:
        """
        Set the label displayed angle.
        Once done, call updateMin().

        Parameters
        ----------
        angle
            angle of the displayed text, by default 0
        """

        self.angle = angle
        self.item.resetTransform()
        self.item.setRotation(angle)
        self.updateMin()


    def updateMin(self) -> None:
        """
        Update the size of the item.
        Once done, call updateGeometry().
        """
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
                       constraint: QtCore.QSizeF) -> QtCore.QSizeF:
        """
        Return the size of the widget following the hint argument.

        Parameters
        ----------
        hint:
            Integer used to specify the minimum size of a graphics layout item:
            0 : MinimumSize
            1 : PreferredSize
            2 : MaximumSize
            3 : MinimumDescent
        constraint:
            no idea, not used
        """

        if hint not in self._sizeHint:
            return QtCore.QSizeF(0, 0)
        return QtCore.QSizeF(*self._sizeHint[hint])


    def itemRect(self) -> QtCore.QRectF:
        """
        Return the item pyqt rectangle coordinates
        """

        return self.item.mapRectToParent(self.item.boundingRect())
