from typing import Union
from pathlib import Path

import importlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import OrderedDict
from qtpy import QtGui
from qtpy.QtCore import QDateTime
from pymodaq_gui.parameter import Parameter

from pyqtgraph.parametertree.Parameter import PARAM_TYPES, PARAM_NAMES

from pymodaq_gui.parameter.xml_parameter_factory import XMLParameterFactory,XMLParameter


def dict_from_param(param):
    """Get Parameter properties as a dictionary

    Parameters
    ----------
    param: Parameter

    Returns
    -------
    opts: dict

    See Also
    --------
    add_text_to_elt, walk_parameters_to_xml, dict_from_param
    """

    param_type = str(param.type())
    
    param_class = XMLParameterFactory.get_parameter_class(param_type)

    opts = param_class.get_options(param)
    
    return opts


# def elt_to_dict(el):
#     """Convert xml element attributes to a dictionnary

#     Parameters
#     ----------
#     el

#     Returns
#     -------

#     """

#     param_type = el.get('type')
    
#     param_instance = XMLParameterFactory.get_parameter_class(param_type)
    
#     return param_instance.xml_elt_to_dict(el)

def set_dict_from_el(el):
    """Convert an element into a dict
    ----------
    el: xml element
    param_dict: dictionnary from which the parameter will be constructed
    """
    param_dict = XMLParameter.xml_elt_to_dict(el)
    return param_dict 

# def parameter_to_xml_string(parent_elt=None, param=None):
#     """ Convert  a Parameter to a XML string.

#     Parameters
#     ----------
#     param: Parameter

#     Returns
#     -------
#     str: XMl string

#     See Also
#     --------
#     add_text_to_elt, walk_parameters_to_xml, dict_from_param

#     Examples
#     --------
#     >>> from pyqtgraph.parametertree import Parameter
#     >>>    #Create an instance of Parameter
#     >>> settings=Parameter(name='settings')
#     >>> converted_xml=parameter_to_xml_string(settings)
#     >>>    # The converted Parameter
#     >>> print(converted_xml)
#     b'<settings title="settings" type="None" />'
#     """
#     xml_elt = XMLParameterFactory.parameter_to_xml_string(parent_elt, param)
#     return xml_elt


def parameter_to_xml_string(param):
    """ Convert  a Parameter to a XML string.

    Parameters
    ----------
    param: Parameter

    Returns
    -------
    str: XMl string

    See Also
    --------
    add_text_to_elt, walk_parameters_to_xml, dict_from_param

    Examples
    --------
    >>> from pyqtgraph.parametertree import Parameter
    >>>    #Create an instance of Parameter
    >>> settings=Parameter(name='settings')
    >>> converted_xml=parameter_to_xml_string(settings)
    >>>    # The converted Parameter
    >>> print(converted_xml)
    b'<settings title="settings" type="None" />'
    """
    xml_elt = XMLParameterFactory.parameter_to_xml_string_factory(param=param)
    return ET.tostring(xml_elt)

def XML_string_to_parameter(xml_string):
    """
        Convert a xml string into a list of dict for initialize pyqtgraph parameter object.

        =============== =========== ================================
        **Parameters**   **Type**    **Description**

        xml_string       string      the xml string to be converted
        =============== =========== ================================

        Returns
        -------
        params: a parameter list of dict to init a parameter

        See Also
        --------
        walk_parameters_to_xml

        Examples
        --------
    """
    root = ET.fromstring(xml_string)
    tree = ET.ElementTree(root)

    # tree.write('test.xml')
    params = XMLParameterFactory.xml_string_to_parameter_list_factory(params=[],XML_elt=root)

    return params