from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.xml_factory import XMLParameterFactory
import pyqtgraph as pg
from pyqtgraph.Qt.QtGui import QColor

factory = XMLParameterFactory()

pg.mkQApp()


def test_xml_serialize():
    params = [
        {'name': 'p_int', 'type': 'int', 'value': 10, 'title': 'Integer Parameter',
         'visible': True, 'removable': False,},
        {'name': 'p_float', 'type': 'float', 'value': 3.14, 'title': 'Float Parameter',
         'visible': True, 'removable': False, 'readonly': False, 'tip': ''},
        {'name': 'p_str', 'type': 'str', 'value': 'Hello', 'title': 'String Parameter',
         'visible': True, 'removable': False, 'readonly': False, 'tip': ''},
        {'name': 'p_group', 'type': 'group', 'children': [
            {'name': 'p_bool', 'type': 'bool', 'value': True, 'title': 'Boolean Parameter',
             'visible': True, 'removable': False, 'readonly': False, 'tip': ''},
            {'name': 'p_color', 'type': 'color', 'value': QColor(1, 2, 3, 1), 'title': 'Color Parameter',
             'visible': True, 'removable': False, 'readonly': False, 'tip': ''},
        ]},
    ]
    settings = Parameter.create(name='settings', type='group', title='setting test',children=params, visible = True, removable = False, readonly = False, tip = '')

    xml_string = factory.parameter_to_xml_string(param=settings)

    assert isinstance(xml_string, bytes)

    param_list_dict = factory.xml_string_to_parameter_list_dict(xml_string)
    for dict_elt in param_list_dict:
        assert isinstance(dict_elt, dict)

    param_res = factory.parameter_list_to_parameter(param_list_dict)

    assert pg.eq(settings.saveState(), param_res.saveState())
