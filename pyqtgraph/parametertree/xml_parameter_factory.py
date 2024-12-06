from abc import abstractmethod
from xml.etree import ElementTree as ET
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.Parameter import Parameter


class XMLParameter():
    @staticmethod
    def set_basic_options(el: ET.Element):
        basic_options = {
            "name": el.tag,
            "type": el.get('type'),
            "title": el.get('title', el.tag),
            "visible": el.get('visible', '0') == '1',
            "removable": el.get('removable', '0') == '1',
            "readonly": el.get('readonly', '0') == '1',
            "tip": el.get('tip', '0') == '1',
            "show_pb": el.get('show_pb', '0') == '1'

        }
        return basic_options
     
    @staticmethod
    @abstractmethod
    def set_specific_options(el:ET.Element):
        pass

    @staticmethod
    def xml_elt_to_dict(el: ET.Element):
        """Convert XML element to a dictionary."""
        param_type = el.get('type',None)
        param_class = XMLParameterFactory.get_parameter_class(param_type)
        if(param_type == 'group'):
            return param_class.set_basic_options(el)
        dic = param_class.set_basic_options(el)
        dic.update(param_class.set_specific_options(el))
        
        return dic

    @staticmethod
    def get_basics_options(param: 'Parameter'):
        opts = {
            "type": param.opts.get("type"),
        }

        title = param.opts['title']
        if title is None:
            title = param.name()

        opts.update(dict(title=title))

        boolean_opts = {
            "visible": param.opts.get("visible", False),
            "removable": param.opts.get("removable", False),
            "readonly": param.opts.get("readonly", False),
            "tip": param.opts.get("tip", False),
            "show_pb": param.opts.get("show_pb", False),
        }
        
        opts.update({key: '1' if value else '0' for key, value in boolean_opts.items()})

        for key in ["limits", "addList", "addText", "detlist", "movelist", "filetype"]:
            if key in param.opts:
                opts[key] = str(param.opts[key])

        return opts
    
    @staticmethod
    @abstractmethod
    def get_specific_options(param: 'Parameter'):
        pass

    @staticmethod
    def get_options(param: 'Parameter'):
        param_type = param.type()
        param_class = XMLParameterFactory.get_parameter_class(param_type)
        if(param_type == 'group'):
            return param_class.get_basics_options(param)
        else:
            dic = param_class.get_basics_options(param)
            dic.update(param_class.get_specific_options(param))
        return dic


class XMLParameterFactory:
    
    text_adders_registry = PARAM_TYPES

    @classmethod
    def get_parameter_class(cls, param_type: str):
        """
        Retrieve the parameter class associated with a given parameter type.
        Args:
            param_type (str): The type of the parameter to retrieve.
        Returns:
            type: The class associated with the given parameter type.
        Raises:
            ValueError: If the provided parameter type is not supported.
        """
        

        normalized_type = param_type.lower()

        for key, parameter_class in cls.text_adders_registry.items():
            if key.lower() == normalized_type:
                return parameter_class

        raise ValueError(f"{param_type} is not a supported parameter type.")

    
    @staticmethod
    def xml_string_to_parameter_list_factory(params=[], XML_elt=None):
        try:
            if type(XML_elt) is not ET.Element:
                raise TypeError('not valid XML element')

            param_dict = XMLParameter.xml_elt_to_dict(XML_elt)

            if param_dict['type'] == 'group':
                param_dict['children'] = []
                for child in XML_elt:
                    child_params = []
                    children = XMLParameterFactory.xml_string_to_parameter_list_factory(child_params, child)
                    param_dict['children'].extend(children)

            params.append(param_dict)

        except Exception as e:  # Handle exceptions for debugging
            raise e
        return params
    
    @staticmethod
    def parameter_to_xml_string_factory(parent_elt = None, param = None):
        from pymodaq_gui.parameter.ioxml_factory import dict_from_param
        """
        To convert a parameter object (and children) to xml data tree.

        =============== ================================ ==================================
        **Parameters**   **Type**                         **Description**

        *parent_elt*     XML element                      the root element
        *param*          instance of pyqtgraph parameter  Parameter object to be converted
        =============== ================================ ==================================

        Returns
        -------
        XML element : parent_elt
            XML element with subelements from Parameter object

        """
        if type(param) is None:
            raise TypeError('No valid param input')

        if parent_elt is None:
            opts = XMLParameter.get_options(param)
            parent_elt = ET.Element(param.name(), **opts)

        params_list = param.children()
        for param in params_list:
            opts = XMLParameter.get_options(param)
            elt = ET.Element(param.name(), **opts)

            if param.hasChildren():
                XMLParameterFactory.parameter_to_xml_string_factory(elt, param)

            parent_elt.append(elt)

        return parent_elt
    
    @staticmethod
    def parameter_list_to_parameter(params: list):
        """
        Convert a list of dict to a pyqtgraph parameter object.

        =============== =========== ================================
        **Parameters**   **Type**    **Description**

        params           list        list of dict to init a parameter
        =============== =========== ================================

        Returns
        -------
        Parameter: a parameter object

        """
        if type(params) is not list:
            raise TypeError('params must be a list of dict') 
        
        
        param_opts = params[0]
        children = param_opts.pop('children', [])

        param = Parameter.create(**param_opts)

        for child_dict in children:
            child_param = XMLParameterFactory.parameter_list_to_parameter([child_dict])
            param.addChild(child_param)

        return param
