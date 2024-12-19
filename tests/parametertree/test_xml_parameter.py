from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.xml_factory import XMLParameterFactory
import pyqtgraph as pg


factory = XMLParameterFactory()

pg.mkQApp()


def test_xml_serialize():
    params = [
        {'name': 'param1', 'type': 'int', 'value': 10, 'title': 'Integer Parameter',
         'visible': True, 'removable': False, 'readonly': False, 'tip': '', 'show_pb': False},
        {'name': 'param2', 'type': 'float', 'value': 3.14, 'title': 'Float Parameter',
         'visible': True, 'removable': False, 'readonly': False, 'tip': '', 'show_pb': False},
        {'name': 'param3', 'type': 'str', 'value': 'Hello', 'title': 'String Parameter',
         'visible': True, 'removable': False, 'readonly': False, 'tip': '', 'show_pb': False},
        {'name': 'param4', 'type': 'bool', 'value': True, 'title': 'Boolean Parameter',
         'visible': True, 'removable': False, 'readonly': False, 'tip': '', 'show_pb': False}
    ]
    settings = Parameter.create(name='settings', type='group', title='setting test',children=params, visible = True, removable = False, readonly = False, tip = '', show_pb = False)

    xml_string = factory.parameter_to_xml_string(param=settings)

    assert isinstance(xml_string, bytes)

    param_list_dict = factory.xml_string_to_parameter_list_dict(xml_string)
    for dict_elt in param_list_dict:
        assert isinstance(dict_elt, dict)

    param_res = factory.parameter_list_to_parameter(param_list_dict)

    assert settings.saveState() == param_res.saveState()
