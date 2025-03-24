import sys

import numpy as np

from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtWidgets
from .Exporter import Exporter

translate = QtCore.QCoreApplication.translate
__all__ = ['ImageExporter']

class ImageExporter(Exporter):
    Name = "Image File (PNG, TIF, JPG, ...)"
    allowCopy = True
    
    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        if isinstance(item, QtWidgets.QGraphicsItem):
            scene = item.scene()
        else:
            scene = item
        bgbrush = scene.views()[0].backgroundBrush()
        bg = bgbrush.color()
        if bgbrush.style() == QtCore.Qt.BrushStyle.NoBrush:
            bg.setAlpha(0)

        self.params = Parameter.create(name='params', type='group', children=[
            {
                'name': 'width',
                'title': translate("Exporter", 'width'),
                'type': 'int',
                'value': int(tr.width()),
                'limits': (0, None)
            },
            {
                'name': 'height',
                'title': translate("Exporter", 'height'),
                'type': 'int',
                'value': int(tr.height()),
                'limits': (0, None)
            },
            {
                'name': 'antialias',
                'title': translate("Exporter", 'antialias'),
                'type': 'bool',
                'value': True
            },
            {
                'name': 'background',
                'title': translate("Exporter", 'background'),
                'type': 'color',
                'value': bg
            },
            {
                'name': 'invertValue',
                'title': translate("Exporter", 'invertValue'),
                'type': 'bool',
                'value': False
            }
        ])
        self.params.param('width').sigValueChanged.connect(self.widthChanged)
        self.params.param('height').sigValueChanged.connect(self.heightChanged)
        
    def widthChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.height()) / sr.width()
        self.params.param('height').setValue(int(self.params['width'] * ar), blockSignal=self.heightChanged)
        
    def heightChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.width()) / sr.height()
        self.params.param('width').setValue(int(self.params['height'] * ar), blockSignal=self.widthChanged)
        
    def parameters(self):
        return self.params

    @staticmethod
    def getSupportedImageFormats():
        filter    = ["*."+f.data().decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
        preferred = ['*.png', '*.tif', '*.jpg']
        for p in preferred[::-1]:
            if p in filter:
                filter.remove(p)
                filter.insert(0, p)
        return filter  

    def export(self, fileName=None, toBytes=False, copy=False):
        if fileName is None and not toBytes and not copy:
            filter = self.getSupportedImageFormats()
            self.fileSaveDialog(filter=filter)
            return

        w = int(self.params['width'])
        h = int(self.params['height'])
        if w == 0 or h == 0:
            raise Exception("Cannot export image with size=0 (requested "
                            "export size is %dx%d)" % (w, h))

        targetRect = QtCore.QRect(0, 0, w, h)
        sourceRect = self.getSourceRect()

        self.png = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        self.png.fill(self.params['background'])
        
        ## set resolution of image:
        origTargetRect = self.getTargetRect()
        resolutionScale = targetRect.width() / origTargetRect.width()
        #self.png.setDotsPerMeterX(self.png.dotsPerMeterX() * resolutionScale)
        #self.png.setDotsPerMeterY(self.png.dotsPerMeterY() * resolutionScale)
        
        painter = QtGui.QPainter(self.png)
        #dtr = painter.deviceTransform()
        try:
            self.setExportMode(True, {
                'antialias': self.params['antialias'],
                'background': self.params['background'],
                'painter': painter,
                'resolutionScale': resolutionScale})
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, self.params['antialias'])
            self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()
        
        if self.params['invertValue']:
            bg = fn.ndarray_from_qimage(self.png)
            if sys.byteorder == 'little':
                cv = slice(0, 3)
            else:
                cv = slice(1, 4)
            mn = bg[...,cv].min(axis=2)
            mx = bg[...,cv].max(axis=2)
            d = (255 - mx) - mn
            bg[...,cv] += d[...,np.newaxis]
        
        if copy:
            QtWidgets.QApplication.clipboard().setImage(self.png)
        elif toBytes:
            return self.png
        else:
            return self.png.save(fileName)
        
ImageExporter.register()
