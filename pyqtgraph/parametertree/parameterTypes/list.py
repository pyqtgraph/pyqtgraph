import warnings
from collections import OrderedDict

from ... import functions as fn
from ...Qt import QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem


class ListParameterItem(WidgetParameterItem):
    """
    WidgetParameterItem subclass providing comboBox that lets the user select from a list of options.

    """
    def __init__(self, param, depth):
        self.targetValue = None
        WidgetParameterItem.__init__(self, param, depth)

    def makeWidget(self):
        w = QtWidgets.QComboBox()
        w.setMaximumHeight(20)  ## set to match height of spin box and line edit
        w.sigChanged = w.currentIndexChanged
        w.value = self.value
        w.setValue = self.setValue
        self.widget = w  ## needs to be set before limits are changed
        self.limitsChanged(self.param, self.param.opts['limits'])
        if len(self.forward) > 0:
            self.setValue(self.param.value())
        return w

    def value(self):
        key = self.widget.currentText()

        return self.forward.get(key, None)

    def setValue(self, val):
        self.targetValue = val
        match = [fn.eq(val, limVal) for limVal in self.reverse[0]]
        if not any(match):
            self.widget.setCurrentIndex(0)
        else:
            idx = match.index(True)
            key = self.reverse[1][idx]
            ind = self.widget.findText(key)
            self.widget.setCurrentIndex(ind)

    def limitsChanged(self, param, limits):
        # set up forward / reverse mappings for name:value

        if len(limits) == 0:
            limits = ['']  ## Can never have an empty list--there is always at least a singhe blank item.

        self.forward, self.reverse = ListParameter.mapping(limits)
        try:
            self.widget.blockSignals(True)
            val = self.targetValue

            self.widget.clear()
            for k in self.forward:
                self.widget.addItem(k)
                if k == val:
                    self.widget.setCurrentIndex(self.widget.count()-1)
                    self.updateDisplayLabel()
        finally:
            self.widget.blockSignals(False)

    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.widget.currentText()
        super().updateDisplayLabel(value)


class ListParameter(Parameter):
    """Parameter with a list of acceptable values.

    By default, this parameter is represtented by a :class:`ListParameterItem`,
    displaying a combo box to select a value from the list.

    In addition to the generic :class:`~pyqtgraph.parametertree.Parameter`
    options, this parameter type accepts a ``limits`` argument specifying the
    list of allowed values.

    The values may generally be of any data type, as long as they can be
    represented as a string. If the string representation provided is
    undesirable, the values may be given as a dictionary mapping the desired
    string representation to the value.
    """

    itemClass = ListParameterItem

    def __init__(self, **opts):
        self.forward = OrderedDict()  ## {name: value, ...}
        self.reverse = ([], [])       ## ([value, ...], [name, ...])

        # Parameter uses 'limits' option to define the set of allowed values
        if 'values' in opts:
            warnings.warn('Using "values" to set limits is deprecated. Use "limits" instead.',
                          DeprecationWarning, stacklevel=2)
            opts['limits'] = opts['values']
        if opts.get('limits', None) is None:
            opts['limits'] = []
        Parameter.__init__(self, **opts)
        self.setLimits(opts['limits'])

    def setLimits(self, limits):
        """Change the list of allowed values."""
        self.forward, self.reverse = self.mapping(limits)

        Parameter.setLimits(self, limits)
        # 'value in limits' expression will break when reverse contains numpy array
        curVal = self.value()
        if len(self.reverse[0]) > 0 and not any(fn.eq(curVal, limVal) for limVal in self.reverse[0]):
            self.setValue(self.reverse[0][0])

    @staticmethod
    def mapping(limits):
        # Return forward and reverse mapping objects given a limit specification
        forward = OrderedDict()  ## {name: value, ...}
        reverse = ([], [])       ## ([value, ...], [name, ...])
        if not isinstance(limits, dict):
            limits = {str(l): l for l in limits}
        for k, v in limits.items():
            forward[k] = v
            reverse[0].append(v)
            reverse[1].append(k)
        return forward, reverse
