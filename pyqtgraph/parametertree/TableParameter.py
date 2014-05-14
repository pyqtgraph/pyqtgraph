import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

import re, sys, os
import numpy

#==================================================================================================
class TableParameterItem(pTypes.WidgetParameterItem):
    """
    A table widget used to edit a numpy record array. 
    
    ================= ==================================================================
    **Arguments:**
    value             (numpy RecArray) initial value of the recArray. It must be provided 
                      to define the size and structure of the table.
    others            any other arguments are passed to *Parameter*
    ================= ==================================================================
    """
    def __init__(self, param, depth):
        pTypes.WidgetParameterItem.__init__(self, param, depth)
        self.hideWidget = False

    def makeWidget(self):
        opts = self.param.opts
        value = opts.get("value", None)
        if value is None:
            raise Exception("Initial value must be provided for TableParameterItem")

        self.dtype = value.dtype
        nr = value.size
        nc = len(self.dtype.names)
        w = QtGui.QTableWidget(nr, nc)
        #w = pg.TableWidget(nr, nc)
        
        hheader = QtGui.QHeaderView(QtCore.Qt.Orientation.Horizontal)
        hheader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        w.setHorizontalHeader(hheader)
        w.setHorizontalHeaderLabels(value.dtype.names)
            
        w.sigChanged = w.cellChanged
        w.value = self.value
        w.setValue = self.setValue

        self.widget = w
        return w
        
    def value(self):
        w = self.widget
        
        nr = w.rowCount()
        arr = numpy.zeros((nr,), dtype=self.dtype)

        try:        
            for ir in xrange(nr):
                for ic, n in enumerate(self.dtype.names):
                    arr[n][ir] =  w.item(ir, ic).text()
        except:
            arr = self.param.opts.get("value")
        
        return arr
            
    def setValue(self, arr):
        nr = arr.size
        w = self.widget
        
        for ir in xrange(nr):
            for ic, n in enumerate(self.dtype.names):
                item = QtGui.QTableWidgetItem(str(arr[n][ir]))
                w.setItem(ir, ic, item)

    def valueChanged(self, param, val, force=False):
        ## called when the parameter's value has changed
        ParameterItem.valueChanged(self, param, val)
        self.widget.sigChanged.disconnect(self.widgetValueChanged)
        try:
            if force or numpy.any(val != self.widget.value()):
                self.widget.setValue(val)
            self.updateDisplayLabel(val)  ## always make sure label is updated, even if values match!
        finally:
            self.widget.sigChanged.connect(self.widgetValueChanged)
        self.updateDefaultBtn()
    
#==================================================================================================
class TableParameter(Parameter):
    itemClass = TableParameterItem

    def __init__(self, **opts):
        Parameter.__init__(self, **opts)

    def setValue(self, value, blockSignal=None):
        """
        Set the value of this Parameter; return the actual value that was set.
        (this may be different from the value that was requested)
        """
        try:
            if blockSignal is not None:
                self.sigValueChanged.disconnect(blockSignal)
            if numpy.all(self.opts['value'] == value):
                return value
            self.opts['value'] = value
            self.sigValueChanged.emit(self, value)
        finally:
            if blockSignal is not None:
                self.sigValueChanged.connect(blockSignal)
            
        return value

    def valueIsDefault(self):
        """Returns True if this parameter's value is equal to the default value."""
        return numpy.all(self.value() == self.defaultValue())

registerParameterType('table', TableParameter, override=True)

# #==================================================================================================
# # Replace original getValues of class Parameter. this new function returns more
# # concise tree of values
# from pyqtgraph.pgcollections import OrderedDict
# def getValues(self):
#     """Return a tree of all values that are children of this parameter"""
#     vals = OrderedDict()
#     for ch in self:
#         if ch.type() == "group":
#             vals[ch.name()] = ch.getValues()
#         else:
#             vals[ch.name()] = ch.value()
#     return vals
# 
# Parameter.getValues = getValues
