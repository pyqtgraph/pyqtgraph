"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types
"""
import io

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

from typing import Optional, List, Type
import random
from xml.etree.ElementTree import ParseError

def print_log(s: str, fp: io.FileIO):
    print(s)
    fp.write(s + '\n')

with open('parameter_serialization_report.txt', 'w') as fp:

    class XMLParameterFactoryModified(XMLParameterFactory):
        def options_from_parameter(self, param: Parameter):
            param_type = param.type()
            if param_type is not None:
                param_class: Type[Parameter] = self.get_parameter_class(param_type)
                dic = {}
                try:
                    dic.update(param_class.shared_options_from_parameter(param))
                except NotImplementedError as e:
                    dic.update({'title': param.name(),
                                'name': param.name(),
                                'type': 'str',
                                'value': 'I"m not implemented so I"m a default string'})
                    print_log(str(e), fp)
                try:
                    dic.update(param_class.specific_options_from_parameter(param))
                except NotImplementedError as e:
                    print_log(str(e), fp)
                except TypeError as e:
                    print_log(str(e), fp)
                return dic
            else:
                print_log(f'Param: {param} has no defined type', fp)
                return {}

    factory = XMLParameterFactoryModified()

    params = makeAllParamTypes()

    ## Create tree of Parameter objects
    p = Parameter.create(name='params', type='group', children=params)


    ## test save/restore
    try:
        xmlState = factory.parameter_to_xml_string(p)
        factory.xml_string_to_parameter_list_dict(xmlState)
    except Exception as e:
        print_log(str(e), fp)

