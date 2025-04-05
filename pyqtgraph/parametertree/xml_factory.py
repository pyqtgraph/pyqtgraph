from typing import Optional, List, Type

from xml.etree import ElementTree as ET

from ..parametertree.Parameter import PARAM_TYPES
from ..parametertree.Parameter import Parameter

    
class XMLParameterFactory:
    """ Object implementing the Factory Pattern to:
    * convert a Parameter object into a serialized xml string
    * convert back a serialized xml string into a Parameter object (or a list of dict to instantiate one)
    * harness the registration mechanism of Parameter types to infer an object from its type
    """
    
    param_types_registry = PARAM_TYPES

    @classmethod
    def get_parameter_class(cls, param_type: str) -> Type[Parameter]:
        """
        Retrieve the parameter class associated with a given parameter type.

        Parameters
        ----------
        param_type: str
            The registered string mapping the type of the parameter to retrieve

        Returns
        -------
        type: The class associated with the given parameter type.

        Raises
        ------
        TypeError: If the provided parameter type is not supported.
        """

        normalized_type = param_type.lower()

        for key, parameter_class in cls.param_types_registry.items():
            if key.lower() == normalized_type:
                return parameter_class

        raise TypeError(f"{param_type} is not a supported parameter type.")

    def options_from_parameter(self, param: Parameter):
        """Convert a Parameter object to a dictionary"""
        param_type = param.type()
        param_class: Type[Parameter] = self.get_parameter_class(param_type)
        
        dic = param_class.shared_options_from_parameter(param)
        dic.update(param_class.specific_options_from_parameter(param))
        
        return dic

    def options_from_xml(self, el: ET.Element):
        """Convert an XML element to a dictionary"""
        param_type = el.get('type')
        param_class: Type[Parameter] = self.get_parameter_class(param_type)

        dic = param_class.shared_options_from_xml(el)
        dic.update(param_class.specific_options_from_xml(el))
        
        return dic

    def xml_elt_to_parameter_list(self, params: List[dict],
                                  xml_elt: ET.Element):
        """ Convert a XML element to a list of dictionaries """

        if not isinstance(xml_elt, ET.Element):
            raise TypeError(f'{xml_elt} is not a valid XML element')

        param_dict = self.options_from_xml(xml_elt)

        if param_dict['type'] == 'group':
            param_dict['children'] = []
            for child in xml_elt:
                child_params = []
                self.xml_elt_to_parameter_list(child_params, child)
                param_dict['children'].extend(child_params)
        params.append(param_dict)

    def parameter_to_xml_string(self, param: Parameter,
                                parent_xml_elt: Optional[ET.Element] = None) -> bytes:
        """ Convert a parameter object (and children) to xml binary string.

        """

        if parent_xml_elt is None:
            opts = self.options_from_parameter(param)
            parent_xml_elt = ET.Element(param.name(), **opts)

        params_list = param.children()
        for param in params_list:
            opts = self.options_from_parameter(param)
            elt = ET.Element(param.name(), **opts)
            if param.hasChildren():
                self.parameter_to_xml_string(param, elt)
            parent_xml_elt.append(elt)

        return ET.tostring(parent_xml_elt)
    
    @staticmethod
    def parameter_list_to_parameter(params: List[dict]) -> Parameter:
        """ Convert a list of dict to a Parameter object.

        Parameters
        ----------
        params: list of dict defining a Parameter object

        Returns
        -------
        Parameter: the retrieved parameter object
        """
        
        param_opts = params[0]
        children = param_opts.pop('children', [])

        param = Parameter.create(**param_opts)

        for child_dict in children:
            child_param = XMLParameterFactory.parameter_list_to_parameter([child_dict])
            param.addChild(child_param)

        return param
    
    def xml_string_to_parameter_list_dict(self, xml_string: bytes) -> List[dict]:
        """ Convert a xml string into a list of dict to initialize a Parameter object.

        Parameters
        ----------
        xml_string: ET.Element dumped into a string
            The binary string to convert to a Parameter

        Returns
        -------
        params: a parameter list of dict to init a parameter
        """
        root = ET.fromstring(xml_string)
        params = []
        self.xml_elt_to_parameter_list(params, xml_elt=root)
        return params

    def xml_string_to_parameter(self, xml_string: bytes) -> Parameter:
        """ Convert a xml string into a Parameter object

        Parameters
        ----------
        xml_string: ET.Element dumped into a string
            The binary string to convert to a Parameter
        Returns
        -------
        Parameter
        """
        return self.parameter_list_to_parameter(self.xml_string_to_parameter_list_dict(xml_string))