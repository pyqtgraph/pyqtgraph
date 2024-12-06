from ...Qt import QtWidgets
from .basetypes import WidgetParameterItem
from pyqtgraph.parametertree.parameterTypes.basetypes import WidgetParameterItem, SimpleParameter, Parameter
from ..xml_parameter_factory import XMLParameter

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


class BoolParameter(Parameter,XMLParameter):
    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','0')
        param_dict['value'] = True if value == '1' else False

        return param_dict
        
    @staticmethod    
    def get_specific_options(param):
        """
        Generate a dictionary of type options for a given parameter of type bool_push.
        Args:
            param (Parameter): The parameter object containing options.
        Returns:
            dict: A dictionary containing the type options for the parameter.
        """

        opts = {
            "value": '1' if param.opts.get('value') == True else '0',
        }

        return opts
    

class BoolPushParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def makeWidget(self):
        opts = self.param.opts
        w = QtWidgets.QPushButton()
        if 'label' in opts:
            w.setText(opts['label'])
        elif 'title' in opts:
            w.setText(opts['title'])
        else:
            w.setText(opts['name'])
        # w.setMaximumWidth(50)
        w.setCheckable(True)
        w.sigChanged = w.toggled
        w.value = w.isChecked
        w.setValue = w.setChecked
        w.setEnabled(not opts.get('readonly', False))
        self.hideWidget = False
        return w


class BoolPushParameter(BoolParameter):
    itemClass = BoolPushParameterItem

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)