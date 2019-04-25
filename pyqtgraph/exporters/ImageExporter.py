from .Exporter import Exporter
from ..parametertree import Parameter
from ..Qt import QtGui, QtCore, QtSvg, QT_LIB
from .. import functions as fn
import numpy as np

__all__ = ['ImageExporter']

class ImageExporter(Exporter):
    Name = "Image File (PNG, TIF, JPG, ...)"
    allowCopy = True
    
    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        if isinstance(item, QtGui.QGraphicsItem):
            scene = item.scene()
        else:
            scene = item
        bgbrush = scene.views()[0].backgroundBrush()
        bg = bgbrush.color()
        if bgbrush.style() == QtCore.Qt.NoBrush:
            bg.setAlpha(0)
            
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'width', 'type': 'int', 'value': int(tr.width()), 'limits': (0, None)},
            {'name': 'height', 'type': 'int', 'value': int(tr.height()), 'limits': (0, None)},
            {'name': 'antialias', 'type': 'bool', 'value': True},
            {'name': 'background', 'type': 'color', 'value': bg},
            {'name': 'invertValue', 'type': 'bool', 'value': False}
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
    
    def export(self, fileName=None, toBytes=False, copy=False):
        if fileName is None and not toBytes and not copy:
            if QT_LIB in ['PySide', 'PySide2']:
                filter = ["*."+str(f) for f in QtGui.QImageWriter.supportedImageFormats()]
            else:
                filter = ["*."+bytes(f).decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
            preferred = ['*.png', '*.tif', '*.jpg']
            for p in preferred[::-1]:
                if p in filter:
                    filter.remove(p)
                    filter.insert(0, p)
            self.fileSaveDialog(filter=filter)
            return
            
        targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
        sourceRect = self.getSourceRect()
        
        
        #self.png = QtGui.QImage(targetRect.size(), QtGui.QImage.Format_ARGB32)
        #self.png.fill(pyqtgraph.mkColor(self.params['background']))
        w, h = self.params['width'], self.params['height']
        if w == 0 or h == 0:
            raise Exception("Cannot export image with size=0 (requested export size is %dx%d)" % (w,h))
        bg = np.empty((self.params['height'], self.params['width'], 4), dtype=np.ubyte)
        color = self.params['background']
        bg[:,:,0] = color.blue()
        bg[:,:,1] = color.green()
        bg[:,:,2] = color.red()
        bg[:,:,3] = color.alpha()

        self.png = fn.makeQImage(bg, alpha=True, copy=False, transpose=False)
        self.bg = bg
        
        ## set resolution of image:
        origTargetRect = self.getTargetRect()
        resolutionScale = targetRect.width() / origTargetRect.width()
        #self.png.setDotsPerMeterX(self.png.dotsPerMeterX() * resolutionScale)
        #self.png.setDotsPerMeterY(self.png.dotsPerMeterY() * resolutionScale)
        
        painter = QtGui.QPainter(self.png)
        #dtr = painter.deviceTransform()
        try:
            self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background'], 'painter': painter, 'resolutionScale': resolutionScale})
            painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
            self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()
        
        if self.params['invertValue']:
            mn = bg[...,:3].min(axis=2)
            mx = bg[...,:3].max(axis=2)
            d = (255 - mx) - mn
            bg[...,:3] += d[...,np.newaxis]
        
        if copy:
            QtGui.QApplication.clipboard().setImage(self.png)
        elif toBytes:
            return self.png
        else:
            self.png.save(fileName)
        
ImageExporter.register()        
        
