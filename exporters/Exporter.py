from pyqtgraph.widgets.FileDialog import FileDialog
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtSvg
import os
LastExportDirectory = None


class Exporter(object):
    """
    Abstract class used for exporting graphics to file / printer / whatever.
    """    

    def __init__(self, item):
        """
        Initialize with the item to be exported.
        Can be an individual graphics item or a scene.
        """
        object.__init__(self)
        self.item = item
        
    def item(self):
        return self.item
    
    def parameters(self):
        """Return the parameters used to configure this exporter."""
        raise Exception("Abstract method must be overridden in subclass.")
        
    def export(self):
        """"""
        raise Exception("Abstract method must be overridden in subclass.")

    def fileSaveDialog(self, filter=None, opts=None):
        ## Show a file dialog, call self.export(fileName) when finished.
        if opts is None:
            opts = {}
        self.fileDialog = FileDialog()
        self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        if filter is not None:
            if isinstance(filter, basestring):
                self.fileDialog.setNameFilter(filter)
            elif isinstance(filter, list):
                self.fileDialog.setNameFilters(filter)
        global LastExportDirectory
        exportDir = LastExportDirectory
        if exportDir is not None:
            self.fileDialog.setDirectory(exportDir)
        self.fileDialog.show()
        self.fileDialog.opts = opts
        self.fileDialog.fileSelected.connect(self.fileSaveFinished)
        return
        
    def fileSaveFinished(self, fileName):
        fileName = str(fileName)
        global LastExportDirectory
        LastExportDirectory = os.path.split(fileName)[0]
        self.export(fileName=fileName, **self.fileDialog.opts)
        
        
    def getScene(self):
        if isinstance(self.item, pg.GraphicsScene):
            return self.item
        else:
            return self.item.scene()
        
    def getSourceRect(self):
        if isinstance(self.item, pg.GraphicsScene):
            return self.item.getViewWidget().viewRect()
        else:
            return self.item.sceneBoundingRect()
        
    def getTargetRect(self):        
        if isinstance(self.item, pg.GraphicsScene):
            return self.item.getViewWidget().rect()
        else:
            return self.item.mapRectToDevice(self.item.boundingRect())
        
        
            
        
    #def writePs(self, fileName=None, item=None):
        #if fileName is None:
            #self.fileSaveDialog(self.writeSvg, filter="PostScript (*.ps)")
            #return
        #if item is None:
            #item = self
        #printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
        #printer.setOutputFileName(fileName)
        #painter = QtGui.QPainter(printer)
        #self.render(painter)
        #painter.end()
    
    #def writeToPrinter(self):
        #pass
