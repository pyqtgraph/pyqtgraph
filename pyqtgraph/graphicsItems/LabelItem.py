from typing import Any, Dict, Optional, Tuple, Union, TypedDict
import warnings

from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    ConfigKeyHint,
    ConfigValueHint,
    initItemStyle)
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor

__all__ = ['LabelItem']

Number = Union[float, int]

optsHint = TypedDict('optsHint',
                     {'color'     : ConfigColorHint,
                      'fontsize'  : Number,
                      'fontweight': str,
                      'fontstyle' : str,
                      'align'     : str,
                      'angle'     : Number},
                     total=False)
# kwargs are not typed because mypy has not ye included Unpack[Typeddict]

class LabelItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.

    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """


    def __init__(self, text: str=' ',
                       parent: Optional[Any]=None,
                       **kwargs) -> None:
        """
        Item to display text.

        Parameters
        ----------
        text: optional
            Text to be displayed, by default ' '
        parent: optional
            Parent item, by default None
        *kwargs: optional
            style options , see setStyle() for accepted style parameters.
        """

        GraphicsWidget.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.item = QtWidgets.QGraphicsTextItem(self)
        self._sizeHint: Dict[int, Tuple[float, float]] = {}

        # Store style options in opts dict
        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'LabelItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)
        self.setText(text)


    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    # All these methods are called automatically because their name
    # follow the definition in the stylesheet, see setStyle()

    ## Font size
    def setFontsize(self, fontsize: Number) -> None:
        """
        Set the font size.

        Parameters
        ----------
        fontsize : float or int
            The font size to set.

        Raises
        ------
        ValueError
            If fontsize is not a float or int.

        Returns
        -------
        None
        """
        if not isinstance(fontsize, float) and not isinstance(fontsize, int):
            raise ValueError('fontsize argument:{} is not a float or a int'.format(fontsize))

        self.opts['fontsize'] = fontsize


    def getFontsize(self) -> Number:
        """
        Get the current font size.

        Returns
        -------
        float or int
            The current font size.
        """
        return self.opts['fontsize']


    ## Font weight
    def setFontweight(self, fontweight: str) -> None:
        """
        Set the font weight.

        Parameters
        ----------
        fontweight : str
            The font weight to set.
            Must be "normal", "bold", "bolder", or "lighter".

        Raises
        ------
        ValueError
            If fontweight is not one of the accepted values.

        Returns
        -------
        None
        """
        if isinstance(fontweight, str):
            if fontweight not in ('normal', 'bold', 'bolder', 'lighter'):
                raise ValueError('fontweight argument:{} must be "normal", "bold", "bolder", or "lighter"'.format(fontweight))
        else:
            raise ValueError('fontweight argument:{} is not a string'.format(fontweight))

        self.opts['fontweight'] = fontweight


    def getFontweight(self) -> str:
        """
        Get the current font weight.

        Returns
        -------
        str
            The current font weight.
        """
        return self.opts['fontweight']

    ## Font style
    def setFontstyle(self, fontstyle: str) -> None:
        """
        Set the font style.

        Parameters
        ----------
        fontstyle : str
            The font style to set.
            Must be "normal", "italic", or "oblique".

        Raises
        ------
        ValueError
            If fontstyle is not one of the accepted values.
        """
        if isinstance(fontstyle, str):
            if fontstyle not in ('normal', 'italic', 'oblique'):
                raise ValueError('fontstyle argument:{} must be "normal", "italic", or "oblique"'.format(fontstyle))
        else:
            raise ValueError('style argument:{} is not a string'.format(fontstyle))

        self.opts['fontstyle'] = fontstyle


    def getFontstyle(self) -> str:
        """
        Get the current font style.

        Returns
        -------
        str
            The current font style.
        """
        return self.opts['fontstyle']

    ## Alignement
    def setAlign(self, align: str) -> None:
        """
        Set the text alignment.

        Parameters
        ----------
        align : str
            The text alignment to set.
            Must be "left", "center", or "right".

        Raises
        ------
        ValueError
            If align is not one of the accepted values.

        Returns
        -------
        None
        """
        if isinstance(align, str):
            if align not in ('left', 'center', 'right'):
                raise ValueError('Given "align" argument:{} must be "left", "center", or "right".'.format(align))
        else:
            raise ValueError('align argument:{} is not a string'.format(align))

        self.opts['align'] = align


    def getAlign(self) -> str:
        """
        Get the current text alignment.

        Returns
        -------
        str
            The current text alignment.
        """
        return self.opts['align']

    ## Color
    def setColor(self, color: ConfigColorHint) -> None:
        """
        Set the color.

        Parameters
        ----------
        color : ConfigColorHint
            The color to set.

        Returns
        -------
        None
        """
        self.opts['color'] = color


    def getColor(self) -> ConfigColorHint:
        """
        Get the current color.

        Returns
        -------
        ConfigColorHint
            The current color.
        """
        return self.opts['color']

    ## Text angle
    def setAngle(self, angle: Number) -> None:
        """
        Set the text angle.

        Parameters
        ----------
        angle : float or int
            The text angle to set.

        Raises
        ------
        ValueError
            If angle is not a float or int.

        Returns
        -------
        None
        """
        if not isinstance(angle, float) and not isinstance(angle, int):
            raise ValueError('angle argument:{} is not a float or int'.format(angle))

        self.opts['angle'] = angle

        # self.angle = angle
        self.item.resetTransform()
        self.item.setRotation(angle)
        self.updateMin()

    def getAngle(self) -> Number:
        """
        Get the current text angle.

        Returns
        -------
        float or int
            The current text angle.
        """
        return self.opts['angle']

    def setStyle(self, **kwargs) -> None:
        """
        Set the style of the LabelItem.

        Parameters
        ----------
        color (ConfigColorHint):
            Text color. Example: '#CCFF00'.
        fontsize (float):
            Text size in pt.
        fontweight {'normal', 'bold', 'bolder' 'lighter'}:
            Text weight.
        fontstyle {'normal', 'italic' 'oblique'}:
            Text style.
        align {'left', 'center', 'right'}:
            Text alignement.
        angle (float):
            Text angle in degrees.

        Deprecated parameters
        ----------
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
            # If the key is a valid entry of the stylesheet
            if k in configStyle['LabelItem'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            # For backward compatibility, we also accept some old key, value tuple
            elif k=='size':
                if isinstance(v, str):
                    self._setSize(v)
                else:
                    raise ValueError('Given size argument:{} is not a string.'.format(v))
            elif k=='bold':
                if isinstance(v, bool):
                    self._setBold(v)
                else:
                    raise ValueError('Given "bold" argument:{}, is not a boolean.'.format(v))
            elif k=='italic':
                if isinstance(v, bool):
                    self._setItalic(v)
                else:
                    raise ValueError('Given "italic" argument:{}, is not a boolean.'.format(v))
            elif k=='justify':
                if isinstance(v, str):
                    self._setJustify(v)
                else:
                    raise ValueError('Given "italic" argument:{}, is not a boolean.'.format(v))
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))

    ##############################################################
    #
    #                   Deprecated style methods
    #
    ##############################################################

    ## bold
    def _setBold(self, bold: bool) -> None:
        """
        Set the font weight to either 'bold' or 'normal' based on the given
        boolean value.

        Parameters
        ----------
        bold : bool
            If True, the font weight will be set to 'bold'. If False, it will
            be set to 'normal'.
        """

        warnings.warn('Argument "bold" is deprecated, "fontweight" should be used instead.',
                       DeprecationWarning,
                       stacklevel=2)

        self.opts['fontweight'] = {True:'bold', False:'normal'}[bold]


    ## size
    def _setSize(self, size: str) -> None:
        """
        Set the font size in points.

        Parameters
        ----------
        size : str
            The font size as a string in the format "xpt", where x is a
            floating number.

        Raises
        ------
        ValueError
            If the given size argument is not in the proper format.
        """

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
        """
        Set the font style to either 'italic' or 'normal' based on the given boolean value.

        Parameters
        ----------
        italic : bool
            If True, the font style will be set to 'italic'. If False, it will be set to 'normal'.
        None
        """

        warnings.warn('Argument "italic" is deprecated, "fontstyle" should be used instead.',
                        DeprecationWarning,
                        stacklevel=2)

        self.opts['fontstyle'] = {True:'italic', False:'normal'}[italic]


    ## justify
    def _setJustify(self, justify: str) -> None:
        """
        Set the text alignment to either 'left', 'center', or 'right'.

        Parameters
        ----------
        justify : str
            The text alignment. Must be one of 'left', 'center', or 'right'.

        Raises
        ------
        ValueError
            If the given justify argument is not one of 'left', 'center', or 'right'.
    """

        warnings.warn('Argument "justify" is deprecated, "align" should be used instead.',
                        DeprecationWarning,
                        stacklevel=2)

        if justify in ('left', 'center', 'right'):
            align = justify
        else:
            raise ValueError('Given "justify" argument:{} must be "left", "center", or "right".'.format(justify))

        self.opts['align'] = align


    def setAttr(self, attr: ConfigKeyHint,
                      value: ConfigValueHint) -> None:
        """
        Deprecated, please use "setStyle".
        Set a style property.

        Parameters
        ----------
        attr : ConfigKeyHint
            The name of the style property to set.
        value : ConfigValueHint
            The value to set the style property to.
        """

        warnings.warn('Method "setAttr" is deprecated. Use "setStyle" instead',
                        DeprecationWarning,
                        stacklevel=2)
        self._setStyle(attr, value)


    ##############################################################
    #
    #                   Text
    #
    ##############################################################


    def setText(self, text: str,
                      **kwargs) -> None:
        """
        Set the text and text properties in the label.

        Parameters
        ----------
        text : str
            The text to be displayed.
        **kwargs : optsHint
            Optional keyword arguments to set text properties:
                - fontsize: font size in points (float)
                - fontweight: font weight as string, either 'normal', 'bold', 'bolder', or 'lighter'.
                - fontstyle: font style as string, either 'normal', 'italic', or 'oblique'.
                - align: text alignment as string, either 'left', 'center', or 'right'.
                - color: color of the text as either a tuple of 3 or 4 integers (RGB or RGBA), or a string
                        representing a named color or a color in hexadecimal format (#RRGGBB or #AARRGGBB).
                - angle: angle of the text in degrees (float).
        """

        self.text = text
        self.setStyle(**kwargs)

        # Most of the style is applied via css
        optlist = []
        optlist.append('color: {}'.format(fn.mkColor(self.opts['color']).name(QtGui.QColor.NameFormat.HexArgb)))
        optlist.append('font-size: {}pt'.format(self.opts['fontsize']))
        optlist.append('font-weight: {}'.format(self.opts['fontweight']))
        optlist.append('font-style: {}'.format(self.opts['fontstyle']))
        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)
        self.item.setHtml(full)

        # The angle style is done directly
        # Note that setAngle  call UpdateMin
        if 'angle' in kwargs:
            self.setAngle(kwargs['angle'])
        else:
            self.setAngle(self.opts['angle'])

        # The alignement is applied  in resizeEvent
        self.resizeEvent(None)
        self.updateGeometry()


    def resizeEvent(self, ev: Optional[QtWidgets.QGraphicsSceneResizeEvent]) -> None:
        """
        Set the position of the LabelItem considering alignement specified either
            in the item via "justify" (deprecated) or "align"
            in the stylesheet via "LabelItem.align"

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
