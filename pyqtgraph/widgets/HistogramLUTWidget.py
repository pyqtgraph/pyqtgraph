# -*- coding: utf-8 -*-
"""
Widget displaying an image histogram along with gradient editor. Can be used to adjust
the appearance of images. This is a wrapper around HistogramLUTItem
"""

from ..Qt import QtGui, QtCore
from .GraphicsView import GraphicsView
from ..graphicsItems.HistogramLUTItem import HistogramLUTItem

__all__ = ['HistogramLUTWidget']


class HistogramLUTWidget(GraphicsView):
    """QWidget wrapper for :class:`~pyqtgraph.HistogramLUTItem`.

    All parameters are passed along in creating the HistogramLUTItem.
    """

    def __init__(self, parent=None, *args, **kargs):
        background = kargs.pop('background', 'default')
        GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
        self.item = HistogramLUTItem(*args, **kargs)
        self.setCentralItem(self.item)

        self.orientation = kargs.get('orientation', 'vertical')
        if self.orientation == 'vertical':
            self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
            self.setMinimumWidth(95)
        else:
            self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
            self.setMinimumHeight(95)

    def sizeHint(self):
        if self.orientation == 'vertical':
            return QtCore.QSize(115, 200)
        else:
            return QtCore.QSize(200, 115)

    def __getattr__(self, attr):
        return getattr(self.item, attr)
