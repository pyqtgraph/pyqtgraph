from ...widgets.SpinBox import SpinBox
from .basetypes import WidgetParameterItem
from .basetypes import SimpleParameter


class NumericParameterItem(WidgetParameterItem):
    """
    Subclasses `WidgetParameterItem` to provide the following types:

    ==========================  =============================================================
    **Registered Types:**
    int                         Displays a :class:`SpinBox <pyqtgraph.SpinBox>` in integer
                                mode.
    float                       Displays a :class:`SpinBox <pyqtgraph.SpinBox>`.
    ==========================  =============================================================
    """
    def makeWidget(self):
        opts = self.param.opts
        t = opts['type']
        defs = {
            'value': 0, 'min': None, 'max': None,
            'step': 1.0, 'dec': False,
            'siPrefix': False, 'suffix': '', 'decimals': 3,
        }
        if t == 'int':
            defs['int'] = True
            defs['minStep'] = 1.0
        for k in defs:
            if k in opts:
                defs[k] = opts[k]
        if opts.get('limits') is not None:
            defs['min'], defs['max'] = opts['limits']
        w = SpinBox()
        w.setOpts(**defs)
        w.sigChanged = w.sigValueChanged
        w.sigChanging = w.sigValueChanging
        return w

    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.widget.lineEdit().text()
        super().updateDisplayLabel(value)

    def showEditor(self):
        super().showEditor()
        self.widget.selectNumber()  # select the numerical portion of the text for quick editing

    def limitsChanged(self, param, limits):
        self.widget.setOpts(bounds=limits)

    def optsChanged(self, param, opts):
        super().optsChanged(param, opts)
        sbOpts = {}
        if 'units' in opts and 'suffix' not in opts:
            sbOpts['suffix'] = opts['units']
        for k, v in opts.items():
            if k in self.widget.opts:
                sbOpts[k] = v
        self.widget.setOpts(**sbOpts)
        self.updateDisplayLabel()


class NumericParameter(SimpleParameter):
    itemClass = NumericParameterItem

    def __init__(self, **opts):
        super().__init__(**opts)

    def setLimits(self, limits):
        curVal = self.value()
        if curVal > limits[1]:
            self.setValue(limits[1])
        elif curVal < limits[0]:
            self.setValue(limits[0])
        super().setLimits(limits)
        return limits
    
    @staticmethod
    def set_specific_options(el):
        value = el.get('value', '0')
        param_dict = {}
        param_type = param_dict['type']

        if param_type == "int":
            param_dict['value'] = int(value)
        elif param_type == "float":
            param_dict['value'] = float(value)

        return param_dict

    def get_specific_options(self):
        if self.opts['type'] == "int":
            value = f'int({self.value()})'
        else:
            value = f'float({self.value()})'

        opts = {
            "value": value,
        }
        return opts
