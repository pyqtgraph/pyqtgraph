import numpy as np

from .ColorMapMenu import ColorMapMenu
from .. import colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['ColorMapButton']


class ColorMapDisplayMixin:
    # Note that this Mixin attempts to be safe for inheritance
    # on either the lhs or rhs. To this end:
    # 1) the __init__ is safe to be called more than once
    # 2) it does not override any main class virtual methods

    def __init__(self, *, orientation):
        self.horizontal = orientation == 'horizontal'
        self._menu = None
        self._setColorMap(None)

    def setMaximumThickness(self, val):
        # calls main class methods
        Thickness = 'Height' if self.horizontal else 'Width'
        getattr(self, f'setMaximum{Thickness}')(val)

    def _setColorMap(self, cmap):
        if isinstance(cmap, str):
            try:
                cmap = colormap.get(cmap)
            except FileNotFoundError:
                cmap = None
        if cmap is None:
            cmap = colormap.ColorMap(None, [0.0, 1.0])

        self._cmap = cmap
        self._image = None

    def setColorMap(self, cmap):
        # calls main class methods
        self._setColorMap(cmap)
        self.colorMapChanged()

    def colorMap(self):
        return self._cmap

    def getImage(self):
        if self._image is None:
            lut = self._cmap.getLookupTable(nPts=256, alpha=True)
            lut = np.expand_dims(lut, axis=0 if self.horizontal else 1)
            qimg = fn.ndarray_to_qimage(lut, QtGui.QImage.Format.Format_RGBA8888)
            self._image = qimg if self.horizontal else qimg.mirrored()
        return self._image

    def getMenu(self):
        if self._menu is None:
            self._menu = ColorMapMenu(showColorMapSubMenus=True)
            self._menu.sigColorMapTriggered.connect(self.setColorMap)
        return self._menu

    def paintColorMap(self, painter, rect):
        painter.save()
        image = self.getImage()
        painter.drawImage(rect, image)

        if not self.horizontal:
            painter.translate(rect.center())
            painter.rotate(-90)
            painter.translate(-rect.center())

        text = self.colorMap().name
        wpen = QtGui.QPen(QtCore.Qt.GlobalColor.white)
        bpen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
        # get an estimate of the lightness of the colormap
        # from its center element
        lightness = image.pixelColor(image.rect().center()).lightnessF()
        if lightness >= 0.5:
            # light: draw text with dark pen
            pens = [wpen, bpen]
        else:
            # dark: draw text with light pen
            pens = [bpen, wpen]

        AF = QtCore.Qt.AlignmentFlag
        trect = painter.boundingRect(rect, AF.AlignCenter, text)
        # draw a background shadow
        painter.setPen(pens[0])
        painter.drawText(trect, 0, text)
        # draw the foreground text
        painter.setPen(pens[1])
        painter.drawText(trect.adjusted(1,0,1,0), 0, text)

        painter.restore()


class ColorMapButton(ColorMapDisplayMixin, QtWidgets.QWidget):
    sigColorMapChanged = QtCore.Signal(object)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        ColorMapDisplayMixin.__init__(self, orientation='horizontal')

    def colorMapChanged(self):
        cmap = self.colorMap()
        self.sigColorMapChanged.emit(cmap)
        self.update()

    def paintEvent(self, evt):
        painter = QtGui.QPainter(self)
        self.paintColorMap(painter, self.contentsRect())
        painter.end()

    def mouseReleaseEvent(self, evt):
        if evt.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        # position the menu below the widget
        pos = self.mapToGlobal(self.pos())
        pos.setY(pos.y() + self.height())
        self.getMenu().popup(pos)
