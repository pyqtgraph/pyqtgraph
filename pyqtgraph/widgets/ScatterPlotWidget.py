from pyqtgraph.Qt import QtGui, QtCore
from .PlotWidget import PlotWidget
from .DataFilterWidget import DataFilterParameter
from .ColorMapWidget import ColorMapParameter
import pyqtgraph.parametertree as ptree
import pyqtgraph.functions as fn
import numpy as np
from pyqtgraph.pgcollections import OrderedDict

__all__ = ['ScatterPlotWidget']

class ScatterPlotWidget(QtGui.QSplitter):
    """
    Given a record array, display a scatter plot of a specific set of data.
    This widget includes controls for selecting the columns to plot,
    filtering data, and determining symbol color and shape. This widget allows
    the user to explore relationships between columns in a record array.
    
    The widget consists of four components:
    
    1) A list of column names from which the user may select 1 or 2 columns
       to plot. If one column is selected, the data for that column will be
       plotted in a histogram-like manner by using :func:`pseudoScatter()
       <pyqtgraph.pseudoScatter>`. If two columns are selected, then the
       scatter plot will be generated with x determined by the first column
       that was selected and y by the second.
    2) A DataFilter that allows the user to select a subset of the data by 
       specifying multiple selection criteria.
    3) A ColorMap that allows the user to determine how points are colored by
       specifying multiple criteria.
    4) A PlotWidget for displaying the data.
    """
    def __init__(self, parent=None):
        QtGui.QSplitter.__init__(self, QtCore.Qt.Horizontal)
        self.ctrlPanel = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.addWidget(self.ctrlPanel)
        self.fieldList = QtGui.QListWidget()
        self.fieldList.setSelectionMode(self.fieldList.ExtendedSelection)
        self.ptree = ptree.ParameterTree(showHeader=False)
        self.filter = DataFilterParameter()
        self.colorMap = ColorMapParameter()
        self.params = ptree.Parameter.create(name='params', type='group', children=[self.filter, self.colorMap])
        self.ptree.setParameters(self.params, showTop=False)
        
        self.plot = PlotWidget()
        self.ctrlPanel.addWidget(self.fieldList)
        self.ctrlPanel.addWidget(self.ptree)
        self.addWidget(self.plot)
        
        self.data = None
        self.style = dict(pen=None, symbol='o')
        
        self.fieldList.itemSelectionChanged.connect(self.fieldSelectionChanged)
        self.filter.sigFilterChanged.connect(self.filterChanged)
        self.colorMap.sigColorMapChanged.connect(self.updatePlot)
    
    def setFields(self, fields):
        """
        Set the list of field names/units to be processed.
        Format is: [(name, units), ...]   
        """
        self.fields = OrderedDict(fields)
        self.fieldList.clear()
        for f,opts in fields:
            item = QtGui.QListWidgetItem(f)
            item.opts = opts
            item = self.fieldList.addItem(item)
        self.filter.setFields(fields)
        self.colorMap.setFields(fields)
        
    def setData(self, data):
        """
        Set the data to be processed and displayed. 
        Argument must be a numpy record array.
        """
        self.data = data
        self.filtered = None
        self.updatePlot()
        
    def fieldSelectionChanged(self):
        sel = self.fieldList.selectedItems()
        if len(sel) > 2:
            self.fieldList.blockSignals(True)
            try:
                for item in sel[1:-1]:
                    item.setSelected(False)
            finally:
                self.fieldList.blockSignals(False)
                
        self.updatePlot()
        
    def filterChanged(self, f):
        self.filtered = None
        self.updatePlot()
        
    def updatePlot(self):
        self.plot.clear()
        if self.data is None:
            return
        
        if self.filtered is None:
            self.filtered = self.filter.filterData(self.data)
        data = self.filtered
        if len(data) == 0:
            return
        
        colors = np.array([fn.mkBrush(*x) for x in self.colorMap.map(data)])
        
        style = self.style.copy()
        
        ## Look up selected columns and units
        sel = list([str(item.text()) for item in self.fieldList.selectedItems()])
        units = list([item.opts.get('units', '') for item in self.fieldList.selectedItems()])
        if len(sel) == 0:
            self.plot.setTitle('')
            return
        

        if len(sel) == 1:
            self.plot.setLabels(left=('N', ''), bottom=(sel[0], units[0]), title='')
            if len(data) == 0:
                return
            x = data[sel[0]]
            #if x.dtype.kind == 'f':
                #mask = ~np.isnan(x)
            #else:
                #mask = np.ones(len(x), dtype=bool)
            #x = x[mask]
            #style['symbolBrush'] = colors[mask]
            y = None
        elif len(sel) == 2:
            self.plot.setLabels(left=(sel[1],units[1]), bottom=(sel[0],units[0]))
            if len(data) == 0:
                return
            
            xydata = []
            for ax in [0,1]:
                d = data[sel[ax]]
                ## scatter catecorical values just a bit so they show up better in the scatter plot.
                #if sel[ax] in ['MorphologyBSMean', 'MorphologyTDMean', 'FIType']:
                    #d += np.random.normal(size=len(cells), scale=0.1)
                xydata.append(d)
            x,y = xydata
            #mask = np.ones(len(x), dtype=bool)
            #if x.dtype.kind == 'f':
                #mask |= ~np.isnan(x)
            #if y.dtype.kind == 'f':
                #mask |= ~np.isnan(y)
            #x = x[mask]
            #y = y[mask]
            #style['symbolBrush'] = colors[mask]

        ## convert enum-type fields to float, set axis labels
        xy = [x,y]
        for i in [0,1]:
            axis = self.plot.getAxis(['bottom', 'left'][i])
            if xy[i] is not None and xy[i].dtype.kind in ('S', 'O'):
                vals = self.fields[sel[i]].get('values', list(set(xy[i])))
                xy[i] = np.array([vals.index(x) if x in vals else None for x in xy[i]], dtype=float)
                axis.setTicks([list(enumerate(vals))])
            else:
                axis.setTicks(None)  # reset to automatic ticking
        x,y = xy
        
        ## mask out any nan values
        mask = np.ones(len(x), dtype=bool)
        if x.dtype.kind == 'f':
            mask &= ~np.isnan(x)
        if y is not None and y.dtype.kind == 'f':
            mask &= ~np.isnan(y)
        x = x[mask]
        style['symbolBrush'] = colors[mask]

        ## Scatter y-values for a histogram-like appearance
        if y is None:
            y = fn.pseudoScatter(x)
        else:
            y = y[mask]
                
            
        self.plot.plot(x, y, **style)
        
        
