import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("Parameter Tree Example")
from pyqtgraph.parametertree import Parameter, ParameterTree

from pyqtgraph.parametertree.xml_factory import XMLParameterFactory

factory = XMLParameterFactory()

params = [
    {'title': 'Integer Parameter', 'name': 'param1', 'type': 'int', 'value': 10, },
    {'title': 'Float Parameter', 'name': 'param2', 'type': 'float', 'value': 3.14, },
    {'title': 'String Parameter', 'name': 'param3', 'type': 'str', 'value': 'Hello', },
    {'title': 'Boolean Parameter', 'name': 'param4', 'type': 'bool', 'value': True, }
]

p2 = Parameter.create(name='settings', type='group', title='setting test',children=params)


def save_to_xml():
    global xml_data
    xml_data = factory.parameter_to_xml_string(p2)
    print("Export XML:\n", xml_data)


def load_from_xml():
    global xml_data

    if xml_data:
        param_list_dict = factory.xml_string_to_parameter_list_dict(xml_data)
        restored_params = factory.parameter_list_to_parameter(param_list_dict)
        t.setParameters(restored_params, showTop=False)
        print("Import XML completed")


btn_export = QtWidgets.QPushButton("Export to XML")
btn_import = QtWidgets.QPushButton("Import from XML")
btn_export.clicked.connect(save_to_xml)
btn_import.clicked.connect(load_from_xml)

t = ParameterTree()
t.setParameters(p2, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter <-> XML Conversion')

win = QtWidgets.QWidget()
layout = QtWidgets.QGridLayout()
win.setLayout(layout)
layout.addWidget(QtWidgets.QLabel("Use this tool to serialize Parameter objects to XML or deserialize XML into Parameters."), 0,  0, 1, 2)
layout.addWidget(t, 1, 0, 1, 2)
layout.addWidget(btn_export, 2, 0)
layout.addWidget(btn_import, 2, 1)
win.show()


if __name__ == '__main__':
    xml_data = None
    pg.exec()
