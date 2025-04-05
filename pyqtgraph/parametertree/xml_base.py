from abc import abstractmethod
from typing import Union

from xml.etree import ElementTree as ET

from ..parametertree import Parameter


def add_in_dict_if_present(dict_to_populate: dict,
                           obj_to_read: Union[dict, ET.Element],
                           key: str,
                           default_value=None):
    """ Add key, value in dict_to_populate if key in obj_to_read

    If default_value is not None and the key doesn't exist in dict_to_read, use it in
    ict_to_populate
    """
    if isinstance(obj_to_read, ET.Element):
        obj_to_read = obj_to_read.attrib
    if key in obj_to_read:
        if obj_to_read.get(key) is not None:
            dict_to_populate[key] = obj_to_read.get(key)
    elif default_value is not None:
        dict_to_populate[key] = default_value


class XMLParameter:
    """ Mixin class defining methods to serialize Parameter object to XML string """

    @staticmethod
    def shared_options_from_xml(el: ET.Element) -> dict:
        """ Extract XML element into a dictionary for shared options of a Parameter

        For instance: 'name', 'type', 'title', 'visible' ...

        Parameters
        ----------
        el: ET.Element
            The xml element to extract information from

        Returns
        -------
        dict: the dictionary with the shared options
        """
        basic_options = {}
        add_in_dict_if_present(basic_options, el, 'type')
        add_in_dict_if_present(basic_options, el, 'name')
        add_in_dict_if_present(basic_options, el, 'title')
        add_in_dict_if_present(basic_options, el, 'tip')

        bool_opts = {}
        add_in_dict_if_present(bool_opts, el, 'visible')
        add_in_dict_if_present(bool_opts, el, 'removable')
        add_in_dict_if_present(bool_opts, el, 'readonly')
        basic_options.update({key: True if value == '1' else False for key, value in bool_opts.items()})

        return basic_options

    @staticmethod
    @abstractmethod
    def specific_options_from_xml(el: ET.Element) -> dict:
        """ Convert a binary xml string into a dictionary to be used to generate
         a Parameter object

        Parameters
        ----------
        el: ET.Element

        Returns
        -------
        dict
        """
        pass

    @staticmethod
    def shared_options_from_parameter(param: 'Parameter') -> dict:
        """ Extract shared Parameter options into a dictionary string serializable

        For instance will convert the builtins boolean as the string: '1' or '0'

        Shared options are for instance: 'name', 'type', 'title', 'visible' ...

        Parameters
        ----------
        param: Parameter
            The Parameter object to extract shared options from

        Returns
        -------
        dict: the dictionary with the shared options
        """
        opts_to_read = param.opts

        opts = {
            "type": opts_to_read.get("type"),
            "name": opts_to_read.get("name"),
        }
        add_in_dict_if_present(opts, opts_to_read, 'title')
        add_in_dict_if_present(opts, opts_to_read, 'tip')


        boolean_opts = {}
        add_in_dict_if_present(boolean_opts, opts_to_read, 'visible')
        add_in_dict_if_present(boolean_opts, opts_to_read, 'removable')
        add_in_dict_if_present(boolean_opts, opts_to_read, 'readonly')

        opts.update({key: '1' if value else '0' for key, value in boolean_opts.items()})

        return opts

    @staticmethod
    @abstractmethod
    def specific_options_from_parameter(param: 'Parameter') -> dict:
        """ Get the object options specific to its type: value, limits, ...

        Returns
        -------
        dict: dictionary of options
        """
        pass
