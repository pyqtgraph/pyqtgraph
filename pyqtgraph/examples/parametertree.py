"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types
"""

# `makeAllParamTypes` creates several parameters from a dictionary of config specs.
# This contains information about the options for each parameter so they can be directly
# inserted into the example parameter tree. To create your own parameters, simply follow
# the guidelines demonstrated by other parameters created here.
from _buildParamTypes import makeAllParamTypes

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("Parameter Tree Example")
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.xml_factory import XMLParameterFactory
from pyqtgraph.parametertree.interactive import interact


## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class ComplexParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.addChild({'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 'siPrefix': True})
        self.addChild({'name': 'B = 1/A', 'type': 'float', 'value': 1/7., 'suffix': 's', 'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)

    def aChanged(self):
        self.b.setValue(1.0 / self.a.value(), blockSignal=self.bChanged)

    def bChanged(self):
        self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)


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


# [
#         {'name': 'Save State to JSON', 'type': 'action'},
#         {'name': 'Restore State from JSON', 'type': 'action', 'children': [
#             {'name': 'Add missing items', 'type': 'bool', 'value': True},
#             {'name': 'Remove extra items', 'type': 'bool', 'value': True},
#         ]},
#         {'name': 'Save State to XML', 'type': 'action'},
#         {'name': 'Restore State from XML', 'type': 'action'},
#     ]

params = [
    makeAllParamTypes(),
    {'name': 'Save/Restore functionality', 'type': 'group'},
    {'name': 'Custom context menu', 'type': 'group', 'children': [
        {'name': 'List contextMenu', 'type': 'float', 'value': 0, 'context': [
            'menu1',
            'menu2'
        ]},
        {'name': 'Dict contextMenu', 'type': 'float', 'value': 0, 'context': {
            'changeName': 'Title',
            'internal': 'What the user sees',
        }},
    ]},
    ComplexParameter(name='Custom parameter group (reciprocal values)'),
    ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
        {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
        {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
    ]),
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)
saveRestoreParam = p.child('Save/Restore functionality')

## If anything changes in the tree, print a message
def change(param, changes):
    print("tree changes:")
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s'% childName)
        print('  change:    %s'% change)
        print('  data:      %s'% str(data))
        print('  ----------')

p.sigTreeStateChanged.connect(change)


def valueChanging(param, value):
    print("Value changing (not finalized): %s %s" % (param, value))

# Only listen for changes of the 'widget' child:
for child in p.child('Example Parameters'):
    if 'widget' in child.names:
        child.child('widget').sigValueChanging.connect(valueChanging)

@interact.decorate(titleFormat='Save to JSON', parent=saveRestoreParam)
def saveToJson():
    global jsonState
    jsonState = p.saveState()

@interact.decorate(titleFormat='Restore from JSON', parent=saveRestoreParam)
def restoreFromJson(addMissingItems=True, removeExtraItems=True):
    global jsonState
    p.restoreState(jsonState, addChildren=addMissingItems, removeChildren=removeExtraItems)


factory = XMLParameterFactory()
@interact.decorate(titleFormat='Save to XML', parent=saveRestoreParam)
def saveToXml():
    global xmlState
    xmlState = factory.parameter_to_xml_string(p)


@interact.decorate(titleFormat='Restore from XML', parent=saveRestoreParam)
def restoreFromXml():
    global xmlState

    if xmlState:
        paramListDict = factory.xml_string_to_parameter_list_dict(xmlState)
        restoredParams = factory.parameter_list_to_parameter(paramListDict)
        t.setParameters(restoredParams, showTop=False)


## Create two ParameterTree widgets, both accessing the same data
t = ParameterTree()
t.setParameters(p, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter Tree')
t2 = ParameterTree()
t2.setParameters(p, showTop=False)

win = QtWidgets.QWidget()
layout = QtWidgets.QGridLayout()
win.setLayout(layout)
layout.addWidget(QtWidgets.QLabel("These are two views of the same data. They should always display the same values."), 0,  0, 1, 2)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(t2, 1, 1, 1, 1)
win.show()

## test save/restore
jsonState = p.saveState()
#xmlState = factory.parameter_to_xml_string(p)

p.restoreState(jsonState)
compareState = p.saveState()
assert pg.eq(compareState, jsonState)

#p.restoreState(factory.xml_string_to_parameter_list_dict(xmlState))
#compareState = p.saveState()
#assert pg.eq(compareState, factory.xml_string_to_parameter_list_dict(xmlState))

if __name__ == '__main__':
    pg.exec()
