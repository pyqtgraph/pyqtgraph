from ...Qt import QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem


class ProgressBarParameterItem(WidgetParameterItem):
    def makeWidget(self):
        w = QtWidgets.QProgressBar()
        w.setMaximumHeight(20)
        w.sigChanged = w.valueChanged
        self.hideWidget = False
        return w


class ProgressBarParameter(Parameter):
    """
    Displays a progress bar whose value can be set between 0 and 100
    """
    itemClass = ProgressBarParameterItem
