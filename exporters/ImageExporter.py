from Exporter import Exporter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtGui, QtCore, QtSvg
import pyqtgraph as pg
import numpy as np

__all__ = ['ImageExporter']

class ImageExporter(Exporter):
    Name = "Image File (PNG, TIF, JPG, ...)"
    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'width', 'type': 'int', 'value': tr.width(), 'limits': (0, None)},
            {'name': 'height', 'type': 'int', 'value': tr.height(), 'limits': (0, None)},
            {'name': 'antialias', 'type': 'bool', 'value': True},
            {'name': 'background', 'type': 'color', 'value': (0,0,0,255)},
        ])
        self.params.param('width').sigValueChanged.connect(self.widthChanged)
        self.params.param('height').sigValueChanged.connect(self.heightChanged)
        
    def widthChanged(self):
        sr = self.getSourceRect()
        ar = sr.height() / sr.width()
        self.params.param('height').setValue(self.params['width'] * ar, blockSignal=self.heightChanged)
        
    def heightChanged(self):
        sr = self.getSourceRect()
        ar = sr.width() / sr.height()
        self.params.param('width').setValue(self.params['height'] * ar, blockSignal=self.widthChanged)
        
    def parameters(self):
        return self.params
    
    def export(self, fileName=None):
        if fileName is None:
            filter = ["*."+str(f) for f in QtGui.QImageWriter.supportedImageFormats()]
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
        bg = np.empty((self.params['width'], self.params['height'], 4), dtype=np.ubyte)
        color = self.params['background']
        bg[:,:,0] = color.blue()
        bg[:,:,1] = color.green()
        bg[:,:,2] = color.red()
        bg[:,:,3] = color.alpha()
        self.png = pg.makeQImage(bg, alpha=True)
        painter = QtGui.QPainter(self.png)
        try:
            self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background']})
            painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
            self.getScene().render(painter, QtCore.QRectF(targetRect), sourceRect)
        finally:
            self.setExportMode(False)
        self.png.save(fileName)
        painter.end()
        
        