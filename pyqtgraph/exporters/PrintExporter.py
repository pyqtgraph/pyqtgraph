from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtWidgets
from .Exporter import Exporter

translate = QtCore.QCoreApplication.translate

__all__ = ['PrintExporter']  
#__all__ = []   ## Printer is disabled for now--does not work very well.

class PrintExporter(Exporter):
    Name = "Printer"
    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        self.params = Parameter.create(name='params', type='group', children=[
            {'name': 'width', 'title': translate("Exporter", 'width'), 'type': 'float', 'value': 0.1,
             'limits': (0, None), 'suffix': 'm', 'siPrefix': True},
            {'name': 'height', 'title': translate("Exporter", 'height'), 'type': 'float',
             'value': (0.1 * tr.height()) / tr.width(), 'limits': (0, None), 'suffix': 'm', 'siPrefix': True},
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
        printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
        dialog = QtGui.QPrintDialog(printer)
        dialog.setWindowTitle(translate('Exporter', "Print Document"))
        if dialog.exec_() != QtWidgets.QDialog.DialogCode.Accepted:
            return
            
        res = QtGui.QGuiApplication.primaryScreen().physicalDotsPerInchX()
        printer.setResolution(res)
        rect = printer.pageRect()
        center = rect.center()
        h = self.params['height'] * res * 100. / 2.54
        w = self.params['width'] * res * 100. / 2.54
        x = center.x() - w/2.
        y = center.y() - h/2.
        
        targetRect = QtCore.QRect(x, y, w, h)
        sourceRect = self.getSourceRect()
        painter = QtGui.QPainter(printer)
        try:
            self.setExportMode(True, {'painter': painter})
            self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()


#PrintExporter.register()        
