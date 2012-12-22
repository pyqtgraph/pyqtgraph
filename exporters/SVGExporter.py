from .Exporter import Exporter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtGui, QtCore, QtSvg
import re

__all__ = ['SVGExporter']

class SVGExporter(Exporter):
    Name = "Scalable Vector Graphics (SVG)"
    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'width', 'type': 'float', 'value': tr.width(), 'limits': (0, None)},
            {'name': 'height', 'type': 'float', 'value': tr.height(), 'limits': (0, None)},
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
            self.fileSaveDialog(filter="Scalable Vector Graphics (*.svg)")
            return
        self.svg = QtSvg.QSvgGenerator()
        self.svg.setFileName(fileName)
        dpi = QtGui.QDesktopWidget().physicalDpiX()
        ## not really sure why this works, but it seems to be important:
        self.svg.setSize(QtCore.QSize(self.params['width']*dpi/90., self.params['height']*dpi/90.))
        self.svg.setResolution(dpi)
        #self.svg.setViewBox()
        targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
        sourceRect = self.getSourceRect()
        
        painter = QtGui.QPainter(self.svg)
        try:
            self.setExportMode(True)
            self.render(painter, QtCore.QRectF(targetRect), sourceRect)
        finally:
            self.setExportMode(False)
        painter.end()

        ## Workaround to set pen widths correctly
        data = open(fileName).readlines()
        for i in range(len(data)):
            line = data[i]
            m = re.match(r'(<g .*)stroke-width="1"(.*transform="matrix\(([^\)]+)\)".*)', line)
            if m is not None:
                #print "Matched group:", line
                g = m.groups()
                matrix = list(map(float, g[2].split(',')))
                #print "matrix:", matrix
                scale = max(abs(matrix[0]), abs(matrix[3]))
                if scale == 0 or scale == 1.0:
                    continue
                data[i] = g[0] + ' stroke-width="%0.2g" ' % (1.0/scale) + g[1] + '\n'
                #print "old line:", line
                #print "new line:", data[i]
        open(fileName, 'w').write(''.join(data))
