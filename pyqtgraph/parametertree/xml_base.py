from abc import abstractmethod
from xml.etree import ElementTree as ET

from ..parametertree import Parameter


class XMLParameter:
    """ Mixin class defining methods to serialize Parameter object to XML string """

    @staticmethod
    def common_options_from_xml(el: ET.Element) -> dict:
        """ Extract XML element into a dictionary

        Parameters
        ----------
        el: ET.Element
            The xml element to extract information from

        Returns
        -------
        dict
        """
        basic_options = {
            "name": el.get('name'),
            "type": el.get('type'),
            "title": el.get('title', el.tag),
            "visible": el.get('visible', '1') == '1',
            "removable": el.get('removable', '0') == '1',
            "readonly": el.get('readonly', '0') == '1',
            "tip": el.get('tip', ""),
        }

        return basic_options

    @staticmethod
    @abstractmethod
    def specific_options_from_xml(el: ET.Element) -> dict:
        """ Convert a binary xml string into a dictionary to be used to generate a Parameter object

        Parameters
        ----------
        el: ET.Element

        Returns
        -------
        dict
        """
        pass

    def common_options_from_parameter(self: 'Parameter') -> dict:
        opts = {
            "type": self.opts.get("type"),
            "name": self.opts.get("name"),
            "tip": self.opts.get("tip","")
        }

        title = self.opts['title']
        if title is None:
            title = self.name()

        opts.update(dict(title=title))

        boolean_opts = {
            "visible": self.opts.get("visible", True),
            "removable": self.opts.get("removable", False),
            "readonly": self.opts.get("readonly", False),
        }

        opts.update({key: '1' if value else '0' for key, value in boolean_opts.items()})

        return opts

    @abstractmethod
    def specific_options_from_parameter(self: 'Parameter') -> dict:
        """ Get the object options specific to its type: value, limits, ...

        Returns
        -------
        dict: dictionary of options
        """
        pass
