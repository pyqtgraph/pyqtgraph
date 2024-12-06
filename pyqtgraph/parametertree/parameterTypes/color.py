from ... import functions as fn
from ...widgets.ColorButton import ColorButton
from .basetypes import SimpleParameter, WidgetParameterItem
from qtpy import QtGui
from ..xml_parameter_factory import XMLParameter

class ColorParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a :class:`ColorButton <pyqtgraph.ColorButton>` """
    def makeWidget(self):
        w = ColorButton()
        w.sigChanged = w.sigColorChanged
        w.sigChanging = w.sigColorChanging
        w.value = w.color
        w.setValue = w.setColor
        self.hideWidget = False
        w.setFlat(True)
        return w


class ColorParameter(SimpleParameter, XMLParameter):
    itemClass = ColorParameterItem

    def _interpretValue(self, v):
        return fn.mkColor(v)

    def value(self):
        value = super().value()
        if value is None:
            return None
        return fn.mkColor(value)

    def saveState(self, filter=None):
        state = super().saveState(filter)
        state['value'] = self.value().getRgb()
        return state
    
    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','0')
        param_dict['value'] = QtGui.QColor(*eval(value))

        return param_dict
        
    @staticmethod
    def get_specific_options(param):
        param_value = param.opts.get('value', None)
        opts = {
            "value": str([param_value.red(), param_value.green(), param_value.blue(), param_value.alpha()])
        }

        return opts
