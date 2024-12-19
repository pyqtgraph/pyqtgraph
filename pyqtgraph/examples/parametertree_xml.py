import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("Parameter Tree Example")
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree

from xml.etree import ElementTree as ET
from pyqtgraph.parametertree.xml_factory import XMLParameterFactory

factory = XMLParameterFactory()


params = [
    {'title': 'Integer Parameter', 'name': 'param1', 'type': 'int', 'value': 10, },
    {'title': 'Float Parameter', 'name': 'param2', 'type': 'float', 'value': 3.14, },
    {'title': 'String Parameter', 'name': 'param3', 'type': 'str', 'value': 'Hello', },
    {'title': 'Boolean Parameter', 'name': 'param4', 'type': 'bool', 'value': True, }
]

p2 = Parameter.create(name='params', type='group', children=params)


def save_to_xml():
    global xml_data
    xml_data = factory.parameter_to_xml_string(p2)
    print("Export XML:\n", xml_data)


def load_from_xml():
    global xml_data

    if xml_data:
        param_list_dict = factory.xml_string_to_parameter(xml_data)
        restored_params = Parameter.create(name='Restored Parameters', type='group',
                                           children=param_list_dict)
        t.setParameters(restored_params, showTop=False)
        print("Import XML completed")


btn_export = QtWidgets.QPushButton("Export to XML")
btn_import = QtWidgets.QPushButton("Import from XML")
btn_export.clicked.connect(save_to_xml)
btn_import.clicked.connect(load_from_xml)

## Create two ParameterTree widgets, both accessing the same data
t = ParameterTree()
t.setParameters(p2, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter Tree')

win = QtWidgets.QWidget()
layout = QtWidgets.QGridLayout()
win.setLayout(layout)
layout.addWidget(QtWidgets.QLabel("These are two views of the same data. They should always display the same values."), 0,  0, 1, 2)
layout.addWidget(t, 1, 0, 1, 2)
layout.addWidget(btn_export, 2, 0)
layout.addWidget(btn_import, 2, 1)
win.show()


if __name__ == '__main__':
    xml_data = None
    pg.exec()
