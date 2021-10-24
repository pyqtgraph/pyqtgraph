from ...Qt import QtWidgets
from .basetypes import WidgetParameterItem


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
