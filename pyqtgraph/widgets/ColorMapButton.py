import importlib.util
import re

import numpy as np

from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['ColorMapButton']


def buildMenuEntryWidget(cmap, text):
    lut = cmap.getLookupTable(nPts=32, alpha=True)
    qimg = QtGui.QImage(lut, len(lut), 1, QtGui.QImage.Format.Format_RGBA8888)
    pixmap = QtGui.QPixmap.fromImage(qimg)

    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(1,1,1,1)
    label1 = QtWidgets.QLabel()
    label1.setScaledContents(True)
    label1.setPixmap(pixmap)
    label2 = QtWidgets.QLabel(text)
    layout.addWidget(label1, 0)
    layout.addWidget(label2, 1)

    return widget


def preset_gradient_to_colormap(name):
    # generate the hsv two gradients using makeHslCycle
    if name == 'spectrum':
        # steps=30 for 300 degrees gives the same density as
        # the default steps=36 for 360 degrees
        cmap = colormap.makeHslCycle((0, 300/360), steps=30)
    elif name == 'cyclic':
        cmap = colormap.makeHslCycle((1, 0))
    else:
        cmap = colormap.ColorMap(*zip(*Gradients[name]["ticks"]), name=name)
    return cmap


class ColorMapMenu(QtWidgets.QMenu):
    def __init__(self, showGradientSubMenu=False):
        super().__init__()

        topmenu = self
        act = topmenu.addAction('None')
        act.setData((None, None))
        topmenu.addSeparator()

        if showGradientSubMenu:
            submenu = topmenu.addMenu('preset gradient')
            submenu.aboutToShow.connect(self.buildGradientSubMenu)

        submenu = topmenu.addMenu('local')
        submenu.aboutToShow.connect(self.buildLocalSubMenu)

        # render the submenus only if the user actually clicks on it

        have_colorcet = importlib.util.find_spec('colorcet') is not None

        # the local cet files are a subset of the colorcet module.
        # expose just one of them.
        if not have_colorcet:
            submenu = topmenu.addMenu('cet (local)')
            submenu.aboutToShow.connect(self.buildCetLocalSubMenu)
        else:
            submenu = topmenu.addMenu('cet (external)')
            submenu.aboutToShow.connect(self.buildCetExternalSubMenu)

        if importlib.util.find_spec('matplotlib') is not None:
            submenu = topmenu.addMenu('matplotlib')
            submenu.aboutToShow.connect(self.buildMatplotlibSubMenu)

        if have_colorcet:
            submenu = topmenu.addMenu('colorcet')
            submenu.aboutToShow.connect(self.buildColorcetSubMenu)

    def buildGradientSubMenu(self):
        source = 'preset-gradient'
        names = list(Gradients.keys())
        self.buildSubMenu(names, source, sort=False)

    def buildLocalSubMenu(self):
        source = None
        names = colormap.listMaps(source=source)
        names = [x for x in names if not x.startswith('CET')]
        self.buildSubMenu(names, source)

    def buildCetLocalSubMenu(self):
        source = None
        names = colormap.listMaps(source=source)
        names = [x for x in names if x.startswith('CET')]
        self.buildSubMenu(names, source)

    def buildCetExternalSubMenu(self):
        source = 'colorcet'
        names = colormap.listMaps(source=source)
        names = [x for x in names if x.startswith('CET')]
        self.buildSubMenu(names, source)

    def buildMatplotlibSubMenu(self):
        source = 'matplotlib'
        names = colormap.listMaps(source=source)
        # remove entries registered by colorcet
        names = [x for x in names if not x.startswith('cet_')]
        # remove the reversed colormaps
        names = [x for x in names if not x.endswith('_r')]
        self.buildSubMenu(names, source)

    def buildColorcetSubMenu(self):
        # colorcet colormaps with shorter/simpler aliases
        source = 'colorcet'
        import colorcet
        names = list(colorcet.palette_n.keys())
        self.buildSubMenu(names, source)

    def buildSubMenu(self, names, source, sort=True):
        # in Qt6 we could have used Qt.ConnectionType.SingleShotConnection
        menu = self.sender()
        menu.aboutToShow.disconnect()

        if sort:
            pattern = re.compile(r'(\d+)')
            key = lambda x: [int(c) if c.isdigit() else c for c in pattern.split(x)]
            names = sorted(names, key=key)

        for name in names:
            if source == 'preset-gradient':
                cmap = preset_gradient_to_colormap(name)
            else:
                cmap = colormap.get(name, source=source)
            act = QtWidgets.QWidgetAction(menu)
            act.setData((name, source))
            act.setDefaultWidget(buildMenuEntryWidget(cmap, name))
            menu.addAction(act)


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
            self._menu = ColorMapMenu()
            self._menu.triggered.connect(self.menuTriggered)
        return self._menu

    def menuTriggered(self, action):
        name, source = action.data()
        if name is None:
            cmap = None
        elif source == 'preset-gradient':
            cmap = preset_gradient_to_colormap(name)
        else:
            # colormap module maintains a cache keyed by name only.
            # thus if a colormap has the same name in two different sources,
            # we will end up getting whatever was already cached.
            cmap = colormap.get(name, source=source)
        self.setColorMap(cmap)

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
