from .basetypes import GroupParameter
from ..xml_parameter_factory import XMLParameter

class GroupParameter(GroupParameter, XMLParameter):
    @staticmethod
    def get_specific_options(param):
        return {}
    
    @staticmethod
    def set_specific_options(el):
        return {}