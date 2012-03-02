## tests for ParameterTree

## make sure pyqtgraph is in path
import sys,os
md = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(md, '..', '..'))

from pyqtgraph.Qt import QtCore, QtGui
import collections, user
app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType


## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class ComplexParameter(Parameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        Parameter.__init__(self, **opts)
        
        self.addChild({'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 'siPrefix': True})
        self.addChild({'name': 'B = 1/A', 'type': 'float', 'value': 1/7., 'suffix': 's', 'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)
        
    def aChanged(self):
        try:
            self.b.sigValueChanged.disconnect(self.bChanged)
            self.b.setValue(1.0 / self.a.value())
        finally:
            self.b.sigValueChanged.connect(self.bChanged)

    def bChanged(self):
        try:
            self.a.sigValueChanged.disconnect(self.aChanged)
            self.a.setValue(1.0 / self.b.value())
        finally:
            self.a.sigValueChanged.connect(self.aChanged)


## test add/remove
## this group includes a menu allowing the user to add new parameters into its child list
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))


## test column spanning (widget sub-item that spans all columns)
class TextParameterItem(pTypes.WidgetParameterItem):
    def __init__(self, param, depth):
        pTypes.WidgetParameterItem.__init__(self, param, depth)
        self.subItem = QtGui.QTreeWidgetItem()
        self.addChild(self.subItem)

    def treeWidgetChanged(self):
        self.treeWidget().setFirstItemColumnSpanned(self.subItem, True)
        self.treeWidget().setItemWidget(self.subItem, 0, self.textBox)
        self.setExpanded(True)
        
    def makeWidget(self):
        self.textBox = QtGui.QTextEdit()
        self.textBox.setMaximumHeight(100)
        self.textBox.value = lambda: str(self.textBox.toPlainText())
        self.textBox.setValue = self.textBox.setPlainText
        self.textBox.sigChanged = self.textBox.textChanged
        return self.textBox
        
class TextParameter(Parameter):
    type = 'text'
    itemClass = TextParameterItem
    
registerParameterType('text', TextParameter)




params = [
    {'name': 'Group 0', 'type': 'group', 'children': [
        {'name': 'Param 1', 'type': 'int', 'value': 10},
        {'name': 'Param 2', 'type': 'float', 'value': 10},
    ]},
    {'name': 'Group 1', 'type': 'group', 'children': [
        {'name': 'Param 1.1', 'type': 'float', 'value': 1.2e-6, 'dec': True, 'siPrefix': True, 'suffix': 'V'},
        {'name': 'Param 1.2', 'type': 'float', 'value': 1.2e6, 'dec': True, 'siPrefix': True, 'suffix': 'Hz'},
        {'name': 'Group 1.3', 'type': 'group', 'children': [
            {'name': 'Param 1.3.1', 'type': 'int', 'value': 11, 'limits': (-7, 15), 'default': -6},
            {'name': 'Param 1.3.2', 'type': 'float', 'value': 1.2e6, 'dec': True, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
        ]},
        {'name': 'Param 1.4', 'type': 'str', 'value': "hi"},
        {'name': 'Param 1.5', 'type': 'list', 'values': [1,2,3], 'value': 2},
        {'name': 'Param 1.6', 'type': 'list', 'values': {"one": 1, "two": 2, "three": 3}, 'value': 2},
        ComplexParameter(name='ComplexParam'),
        ScalableGroup(name="ScalableGroup", children=[
            {'name': 'ScalableParam 1', 'type': 'str', 'value': "hi"},
            {'name': 'ScalableParam 2', 'type': 'str', 'value': "hi"},
            
        ])
    ]},
    {'name': 'Param 5', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
    {'name': 'Param 6', 'type': 'color', 'value': "FF0", 'tip': "This is a color button. It cam be renamed.", 'renamable': True},
    {'name': 'TextParam', 'type': 'text', 'value': 'Some text...'},
]

#p = pTypes.ParameterSet("params", params)
p = Parameter(name='params', type='group', children=params)
def change(param, changes):
    print "tree changes:"
    for param, change, data in changes:
        print "  [" + '.'.join(p.childPath(param))+ "]   ", change, data
    
p.sigTreeStateChanged.connect(change)


t = ParameterTree()
t.setParameters(p, showTop=False)
t.show()
t.resize(400,600)
t2 = ParameterTree()
t2.setParameters(p, showTop=False)
t2.show()
t2.resize(400,600)
    
import sys
if sys.flags.interactive == 0:
    app.exec_()
