from ...Qt import QtWidgets
from .basetypes import WidgetParameterItem
from pyqtgraph.parametertree.parameterTypes.basetypes import WidgetParameterItem, SimpleParameter, Parameter


class BoolParameterItem(WidgetParameterItem):
    """
    Registered parameter type which displays a QCheckBox
    """
    def makeWidget(self):
        w = QtWidgets.QCheckBox()
        w.sigChanged = w.toggled
        w.value = w.isChecked
        w.setValue = w.setChecked
        self.hideWidget = False
        return w


class BoolParameter(Parameter):
    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','0')
        param_dict['value'] = True if value == '1' else False

        return param_dict
        
    def get_specific_options(self):
        """
        Generate a dictionary of type options for a given parameter of type bool_push.
        Args:
            param (Parameter): The parameter object containing options.
        Returns:
            dict: A dictionary containing the type options for the parameter.
        """

        opts = {
            "value": '1' if self.opts.get('value') is True else '0',
        }

        return opts
