import numpy as np

from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import *


class PlotWidgetNode(Node):
    """Connection to PlotWidget. Will plot arrays, metaarrays, and display event lists."""
    nodeName = 'PlotWidget'
    sigPlotChanged = QtCore.Signal(object)
    
    def __init__(self, name):
        Node.__init__(self, name, terminals={'In': {'io': 'in', 'multi': True}})
        self.plot = None  # currently selected plot 
        self.plots = {}   # list of available plots user may select from
        self.ui = None 
        self.items = {}
        
    def disconnected(self, localTerm, remoteTerm):
        if localTerm is self['In'] and remoteTerm in self.items:
            self.plot.removeItem(self.items[remoteTerm])
            del self.items[remoteTerm]
        
    def setPlot(self, plot):
        #print "======set plot"
        if plot == self.plot:
            return
        
        # clear data from previous plot
        if self.plot is not None:
            for vid in list(self.items.keys()):
                self.plot.removeItem(self.items[vid])
                del self.items[vid]

        self.plot = plot
        self.updateUi()
        self.update()
        self.sigPlotChanged.emit(self)
        
    def getPlot(self):
        return self.plot
        
    def process(self, In, display=True):
        if display and self.plot is not None:
            items = set()
            # Add all new input items to selected plot
            for name, vals in In.items():
                if vals is None:
                    continue
                if type(vals) is not list:
                    vals = [vals]
                    
                for val in vals:
                    vid = id(val)
                    if vid in self.items and self.items[vid].scene() is self.plot.scene():
                        # Item is already added to the correct scene
                        #   possible bug: what if two plots occupy the same scene? (should
                        #   rarely be a problem because items are removed from a plot before
                        #   switching).
                        items.add(vid)
                    else:
                        # Add the item to the plot, or generate a new item if needed.
                        if isinstance(val, QtWidgets.QGraphicsItem):
                            self.plot.addItem(val)
                            item = val
                        else:
                            item = self.plot.plot(val)
                        self.items[vid] = item
                        items.add(vid)
                        
            # Any left-over items that did not appear in the input must be removed
            for vid in list(self.items.keys()):
                if vid not in items:
                    self.plot.removeItem(self.items[vid])
                    del self.items[vid]
            
    def processBypassed(self, args):
        if self.plot is None:
            return
        for item in list(self.items.values()):
            self.plot.removeItem(item)
        self.items = {}
        
    def ctrlWidget(self):
        if self.ui is None:
            self.ui = ComboBox()
            self.ui.currentIndexChanged.connect(self.plotSelected)
            self.updateUi()
        return self.ui
    
    def plotSelected(self, index):
        self.setPlot(self.ui.value())
    
    def setPlotList(self, plots):
        """
        Specify the set of plots (PlotWidget or PlotItem) that the user may
        select from.
        
        *plots* must be a dictionary of {name: plot} pairs.
        """
        self.plots = plots
        self.updateUi()
    
    def updateUi(self):
        # sets list and automatically preserves previous selection
        self.ui.setItems(self.plots)
        try:
            self.ui.setValue(self.plot)
        except ValueError:
            pass
        

class CanvasNode(Node):
    """Connection to a Canvas widget."""
    nodeName = 'CanvasWidget'
    
    def __init__(self, name):
        Node.__init__(self, name, terminals={'In': {'io': 'in', 'multi': True}})
        self.canvas = None
        self.items = {}
        
    def disconnected(self, localTerm, remoteTerm):
        if localTerm is self.In and remoteTerm in self.items:
            self.canvas.removeItem(self.items[remoteTerm])
            del self.items[remoteTerm]
        
    def setCanvas(self, canvas):
        self.canvas = canvas
        
    def getCanvas(self):
        return self.canvas
        
    def process(self, In, display=True):
        if display:
            items = set()
            for name, vals in In.items():
                if vals is None:
                    continue
                if type(vals) is not list:
                    vals = [vals]
                
                for val in vals:
                    vid = id(val)
                    if vid in self.items:
                        items.add(vid)
                    else:
                        self.canvas.addItem(val)
                        item = val
                        self.items[vid] = item
                        items.add(vid)
            for vid in list(self.items.keys()):
                if vid not in items:
                    #print "remove", self.items[vid]
                    self.canvas.removeItem(self.items[vid])
                    del self.items[vid]


class PlotCurve(CtrlNode):
    """Generates a plot curve from x/y data"""
    nodeName = 'PlotCurve'
    uiTemplate = [
        ('color', 'color'),
    ]
    
    def __init__(self, name):
        CtrlNode.__init__(self, name, terminals={
            'x': {'io': 'in'},
            'y': {'io': 'in'},
            'plot': {'io': 'out'}
        })
        self.item = PlotDataItem()
    
    def process(self, x, y, display=True):
        #print "scatterplot process"
        if not display:
            return {'plot': None}
        
        self.item.setData(x, y, pen=self.ctrls['color'].color())
        return {'plot': self.item}
        
        


class ScatterPlot(CtrlNode):
    """Generates a scatter plot from a record array or nested dicts"""
    nodeName = 'ScatterPlot'
    uiTemplate = [
        ('x', 'combo', {'values': [], 'index': 0}),
        ('y', 'combo', {'values': [], 'index': 0}),
        ('sizeEnabled', 'check', {'value': False}),
        ('size', 'combo', {'values': [], 'index': 0}),
        ('absoluteSize', 'check', {'value': False}),
        ('colorEnabled', 'check', {'value': False}),
        ('color', 'colormap', {}),
        ('borderEnabled', 'check', {'value': False}),
        ('border', 'colormap', {}),
    ]
    
    def __init__(self, name):
        CtrlNode.__init__(self, name, terminals={
            'input': {'io': 'in'},
            'plot': {'io': 'out'}
        })
        self.item = ScatterPlotItem()
        self.keys = []
        
        #self.ui = QtWidgets.QWidget()
        #self.layout = QtWidgets.QGridLayout()
        #self.ui.setLayout(self.layout)
        
        #self.xCombo = QtWidgets.QComboBox()
        #self.yCombo = QtWidgets.QComboBox()
        
        
    
    def process(self, input, display=True):
        #print "scatterplot process"
        if not display:
            return {'plot': None}
            
        self.updateKeys(input[0])
        
        x = str(self.ctrls['x'].currentText())
        y = str(self.ctrls['y'].currentText())
        size = str(self.ctrls['size'].currentText())
        pen = QtGui.QPen(QtGui.QColor(0,0,0,0))
        points = []
        for i in input:
            pt = {'pos': (i[x], i[y])}
            if self.ctrls['sizeEnabled'].isChecked():
                pt['size'] = i[size]
            if self.ctrls['borderEnabled'].isChecked():
                pt['pen'] = QtGui.QPen(self.ctrls['border'].getColor(i))
            else:
                pt['pen'] = pen
            if self.ctrls['colorEnabled'].isChecked():
                pt['brush'] = QtGui.QBrush(self.ctrls['color'].getColor(i))
            points.append(pt)
        self.item.setPxMode(not self.ctrls['absoluteSize'].isChecked())
            
        self.item.setPoints(points)
        
        return {'plot': self.item}
        
        

    def updateKeys(self, data):
        if isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list) or isinstance(data, tuple):
            keys = data
        elif isinstance(data, np.ndarray) or isinstance(data, np.void):
            keys = data.dtype.names
        else:
            print("Unknown data type:", type(data), data)
            return
            
        for c in self.ctrls.values():
            c.blockSignals(True)
        for c in [self.ctrls['x'], self.ctrls['y'], self.ctrls['size']]:
            cur = str(c.currentText())
            c.clear()
            for k in keys:
                c.addItem(k)
                if k == cur:
                    c.setCurrentIndex(c.count()-1)
        for c in [self.ctrls['color'], self.ctrls['border']]:
            c.setArgList(keys)
        for c in self.ctrls.values():
            c.blockSignals(False)
                
        self.keys = keys
        

    def saveState(self):
        state = CtrlNode.saveState(self)
        return {'keys': self.keys, 'ctrls': state}
        
    def restoreState(self, state):
        self.updateKeys(state['keys'])
        CtrlNode.restoreState(self, state['ctrls'])
        
#class ImageItem(Node):
    #"""Creates an ImageItem for display in a canvas from a file handle."""
    #nodeName = 'Image'
    
    #def __init__(self, name):
        #Node.__init__(self, name, terminals={
            #'file': {'io': 'in'},
            #'image': {'io': 'out'}
        #})
        #self.imageItem = graphicsItems.ImageItem()
        #self.handle = None
        
    #def process(self, file, display=True):
        #if not display:
            #return {'image': None}
            
        #if file != self.handle:
            #self.handle = file
            #data = file.read()
            #self.imageItem.updateImage(data)
            
        #pos = file.
        
        
        
