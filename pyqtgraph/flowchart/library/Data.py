# -*- coding: utf-8 -*-
from ..Node import Node
from ...Qt import QtGui, QtCore, QtWidgets
import numpy as np
import sys
from .common import *
from ...SRTTransform import SRTTransform
from ...Point import Point
from ...widgets.TreeWidget import TreeWidget
from ...graphicsItems.LinearRegionItem import LinearRegionItem

from . import functions

class ColumnSelectNode(Node):
    """Select named columns from a record array or MetaArray."""
    nodeName = "ColumnSelect"
    def __init__(self, name):
        Node.__init__(self, name, terminals={'In': {'io': 'in'}})
        self.columns = set()
        self.columnList = QtGui.QListWidget()
        self.axis = 0
        self.columnList.itemChanged.connect(self.itemChanged)
        
    def process(self, In, display=True):
        if display:
            self.updateList(In)
                
        out = {}
        if hasattr(In, 'implements') and In.implements('MetaArray'):
            for c in self.columns:
                out[c] = In[self.axis:c]
        elif isinstance(In, np.ndarray) and In.dtype.fields is not None:
            for c in self.columns:
                out[c] = In[c]
        else:
            self.In.setValueAcceptable(False)
            raise Exception("Input must be MetaArray or ndarray with named fields")
            
        return out
        
    def ctrlWidget(self):
        return self.columnList

    def updateList(self, data):
        if hasattr(data, 'implements') and data.implements('MetaArray'):
            cols = data.listColumns()
            for ax in cols:  ## find first axis with columns
                if len(cols[ax]) > 0:
                    self.axis = ax
                    cols = set(cols[ax])
                    break
        else:
            cols = list(data.dtype.fields.keys())
                
        rem = set()
        for c in self.columns:
            if c not in cols:
                self.removeTerminal(c)
                rem.add(c)
        self.columns -= rem
                
        self.columnList.blockSignals(True)
        self.columnList.clear()
        for c in cols:
            item = QtGui.QListWidgetItem(c)
            item.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsUserCheckable)
            if c in self.columns:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
            self.columnList.addItem(item)
        self.columnList.blockSignals(False)
        

    def itemChanged(self, item):
        col = str(item.text())
        if item.checkState() == QtCore.Qt.Checked:
            if col not in self.columns:
                self.columns.add(col)
                self.addOutput(col)
        else:
            if col in self.columns:
                self.columns.remove(col)
                self.removeTerminal(col)
        self.update()
        
    def saveState(self):
        state = Node.saveState(self)
        state['columns'] = list(self.columns)
        return state
    
    def restoreState(self, state):
        Node.restoreState(self, state)
        self.columns = set(state.get('columns', []))
        for c in self.columns:
            self.addOutput(c)



class RegionSelectNode(CtrlNode):
    """Returns a slice from a 1-D array. Connect the 'widget' output to a plot to display a region-selection widget."""
    nodeName = "RegionSelect"
    uiTemplate = [
        ('start', 'spin', {'value': 0, 'step': 0.1}),
        ('stop', 'spin', {'value': 0.1, 'step': 0.1}),
        ('display', 'check', {'value': True}),
        ('movable', 'check', {'value': True}),
    ]
    
    def __init__(self, name):
        self.items = {}
        CtrlNode.__init__(self, name, terminals={
            'data': {'io': 'in'},
            'selected': {'io': 'out'},
            'region': {'io': 'out'},
            'widget': {'io': 'out', 'multi': True}
        })
        self.ctrls['display'].toggled.connect(self.displayToggled)
        self.ctrls['movable'].toggled.connect(self.movableToggled)
        
    def displayToggled(self, b):
        for item in self.items.values():
            item.setVisible(b)
            
    def movableToggled(self, b):
        for item in self.items.values():
            item.setMovable(b)
            
        
    def process(self, data=None, display=True):
        #print "process.."
        s = self.stateGroup.state()
        region = [s['start'], s['stop']]
        
        if display:
            conn = self['widget'].connections()
            for c in conn:
                plot = c.node().getPlot()
                if plot is None:
                    continue
                if c in self.items:
                    item = self.items[c]
                    item.setRegion(region)
                    #print "  set rgn:", c, region
                    #item.setXVals(events)
                else:
                    item = LinearRegionItem(values=region)
                    self.items[c] = item
                    #item.connect(item, QtCore.SIGNAL('regionChanged'), self.rgnChanged)
                    item.sigRegionChanged.connect(self.rgnChanged)
                    item.setVisible(s['display'])
                    item.setMovable(s['movable'])
                    #print "  new rgn:", c, region
                    #self.items[c].setYRange([0., 0.2], relative=True)
        
        if self['selected'].isConnected():
            if data is None:
                sliced = None
            elif (hasattr(data, 'implements') and data.implements('MetaArray')):
                sliced = data[0:s['start']:s['stop']]
            else:
                mask = (data['time'] >= s['start']) * (data['time'] < s['stop'])
            sliced = data[mask]
        else:
            sliced = None
            
        return {'selected': sliced, 'widget': self.items, 'region': region}
        
        
    def rgnChanged(self, item):
        region = item.getRegion()
        self.stateGroup.setState({'start': region[0], 'stop': region[1]})
        self.update()
        
        
class TextEdit(QtWidgets.QTextEdit):
    def __init__(self, on_update):
        super().__init__()
        self.on_update = on_update
        self.lastText = None

    def focusOutEvent(self, ev):
        text = str(self.toPlainText())
        if text != self.lastText:
            self.lastText = text
            self.on_update()
        super().focusOutEvent(ev)


class EvalNode(Node):
    """Return the output of a string evaluated/executed by the python interpreter.
    The string may be either an expression or a python script, and inputs are accessed as the name of the terminal. 
    For expressions, a single value may be evaluated for a single output, or a dict for multiple outputs.
    For a script, the text will be executed as the body of a function."""
    nodeName = 'PythonEval'
    
    def __init__(self, name):
        Node.__init__(self, name, 
            terminals = {
                'input': {'io': 'in', 'renamable': True, 'multiable': True},
                'output': {'io': 'out', 'renamable': True, 'multiable': True},
            },
            allowAddInput=True, allowAddOutput=True)
        
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.text = TextEdit(self.update)
        self.text.setTabStopWidth(30)
        self.text.setPlainText("# Access inputs as args['input_name']\nreturn {'output': None} ## one key per output terminal")
        self.layout.addWidget(self.text, 1, 0, 1, 2)
        self.ui.setLayout(self.layout)
        
    def ctrlWidget(self):
        return self.ui
        
    def setCode(self, code):
        # unindent code; this allows nicer inline code specification when 
        # calling this method.
        ind = []
        lines = code.split('\n')
        for line in lines:
            stripped = line.lstrip()
            if len(stripped) > 0:
                ind.append(len(line) - len(stripped))
        if len(ind) > 0:
            ind = min(ind)
            code = '\n'.join([line[ind:] for line in lines])
        
        self.text.clear()
        self.text.insertPlainText(code)

    def code(self):
        return self.text.toPlainText()
        
    def process(self, display=True, **args):
        l = locals()
        l.update(args)
        ## try eval first, then exec
        try:  
            text = str(self.text.toPlainText()).replace('\n', ' ')
            output = eval(text, globals(), l)
        except SyntaxError:
            fn = "def fn(**args):\n"
            run = "\noutput=fn(**args)\n"
            text = fn + "\n".join(["    "+l for l in str(self.text.toPlainText()).split('\n')]) + run
            if sys.version_info.major == 2:
                exec(text)
            elif sys.version_info.major == 3:
                ldict = locals()
                exec(text, globals(), ldict)
                output = ldict['output']
        except:
            print("Error processing node: %s" % self.name())
            raise
        return output
        
    def saveState(self):
        state = Node.saveState(self)
        state['text'] = str(self.text.toPlainText())
        #state['terminals'] = self.saveTerminals()
        return state
        
    def restoreState(self, state):
        Node.restoreState(self, state)
        self.setCode(state['text'])
        self.restoreTerminals(state['terminals'])
        self.update()

        
class ColumnJoinNode(Node):
    """Concatenates record arrays and/or adds new columns"""
    nodeName = 'ColumnJoin'
    
    def __init__(self, name):
        Node.__init__(self, name, terminals = {
            'output': {'io': 'out'},
        })
        
        #self.items = []
        
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.ui.setLayout(self.layout)
        
        self.tree = TreeWidget()
        self.addInBtn = QtGui.QPushButton('+ Input')
        self.remInBtn = QtGui.QPushButton('- Input')
        
        self.layout.addWidget(self.tree, 0, 0, 1, 2)
        self.layout.addWidget(self.addInBtn, 1, 0)
        self.layout.addWidget(self.remInBtn, 1, 1)

        self.addInBtn.clicked.connect(self.addInput)
        self.remInBtn.clicked.connect(self.remInput)
        self.tree.sigItemMoved.connect(self.update)
        
    def ctrlWidget(self):
        return self.ui
        
    def addInput(self):
        #print "ColumnJoinNode.addInput called."
        term = Node.addInput(self, 'input', renamable=True, removable=True, multiable=True)
        #print "Node.addInput returned. term:", term
        item = QtGui.QTreeWidgetItem([term.name()])
        item.term = term
        term.joinItem = item
        #self.items.append((term, item))
        self.tree.addTopLevelItem(item)

    def remInput(self):
        sel = self.tree.currentItem()
        term = sel.term
        term.joinItem = None
        sel.term = None
        self.tree.removeTopLevelItem(sel)
        self.removeTerminal(term)
        self.update()

    def process(self, display=True, **args):
        order = self.order()
        vals = []
        for name in order:
            if name not in args:
                continue
            val = args[name]
            if isinstance(val, np.ndarray) and len(val.dtype) > 0:
                vals.append(val)
            else:
                vals.append((name, None, val))
        return {'output': functions.concatenateColumns(vals)}

    def order(self):
        return [str(self.tree.topLevelItem(i).text(0)) for i in range(self.tree.topLevelItemCount())]

    def saveState(self):
        state = Node.saveState(self)
        state['order'] = self.order()
        return state
        
    def restoreState(self, state):
        Node.restoreState(self, state)
        inputs = self.inputs()

        ## Node.restoreState should have created all of the terminals we need
        ## However: to maintain support for some older flowchart files, we need
        ## to manually add any terminals that were not taken care of.
        for name in [n for n in state['order'] if n not in inputs]:
            Node.addInput(self, name, renamable=True, removable=True, multiable=True)
        inputs = self.inputs()

        order = [name for name in state['order'] if name in inputs]
        for name in inputs:
            if name not in order:
                order.append(name)
        
        self.tree.clear()
        for name in order:
            term = self[name]
            item = QtGui.QTreeWidgetItem([name])
            item.term = term
            term.joinItem = item
            #self.items.append((term, item))
            self.tree.addTopLevelItem(item)

    def terminalRenamed(self, term, oldName):
        Node.terminalRenamed(self, term, oldName)
        item = term.joinItem
        item.setText(0, term.name())
        self.update()
        
        
class Mean(CtrlNode):
    """Calculate the mean of an array across an axis.
    """
    nodeName = 'Mean'
    uiTemplate = [
        ('axis', 'intSpin', {'value': 0, 'min': -1, 'max': 1000000}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.mean(axis=ax)


class Max(CtrlNode):
    """Calculate the maximum of an array across an axis.
    """
    nodeName = 'Max'
    uiTemplate = [
        ('axis', 'intSpin', {'value': 0, 'min': -1, 'max': 1000000}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.max(axis=ax)


class Min(CtrlNode):
    """Calculate the minimum of an array across an axis.
    """
    nodeName = 'Min'
    uiTemplate = [
        ('axis', 'intSpin', {'value': 0, 'min': -1, 'max': 1000000}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.min(axis=ax)


class Stdev(CtrlNode):
    """Calculate the standard deviation of an array across an axis.
    """
    nodeName = 'Stdev'
    uiTemplate = [
        ('axis', 'intSpin', {'value': -0, 'min': -1, 'max': 1000000}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.std(axis=ax)


class Index(CtrlNode):
    """Select an index from an array axis.
    """
    nodeName = 'Index'
    uiTemplate = [
        ('axis', 'intSpin', {'value': 0, 'min': 0, 'max': 1000000}),
        ('index', 'intSpin', {'value': 0, 'min': 0, 'max': 1000000}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        ax = s['axis']
        ind = s['index']
        if ax == 0:
            # allow support for non-ndarray sequence types
            return data[ind]
        else:
            return data.take(ind, axis=ax)
        

class Slice(CtrlNode):
    """Select a slice from an array axis.
    """
    nodeName = 'Slice'
    uiTemplate = [
        ('axis', 'intSpin', {'value': 0, 'min': 0, 'max': 1e6}),
        ('start', 'intSpin', {'value': 0, 'min': -1e6, 'max': 1e6}),
        ('stop', 'intSpin', {'value': -1, 'min': -1e6, 'max': 1e6}),
        ('step', 'intSpin', {'value': 1, 'min': -1e6, 'max': 1e6}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        ax = s['axis']
        start = s['start']
        stop = s['stop']
        step = s['step']
        if ax == 0:
            # allow support for non-ndarray sequence types
            return data[start:stop:step]
        else:
            sl = [slice(None) for i in range(data.ndim)]
            sl[ax] = slice(start, stop, step)
            return data[sl]
        

class AsType(CtrlNode):
    """Convert an array to a different dtype.
    """
    nodeName = 'AsType'
    uiTemplate = [
        ('dtype', 'combo', {'values': ['float', 'int', 'float32', 'float64', 'float128', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'], 'index': 0}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        return data.astype(s['dtype'])

