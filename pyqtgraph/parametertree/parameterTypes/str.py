from ...Qt import QtWidgets
from .basetypes import WidgetParameterItem


class StrParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def makeWidget(self):
        w = QtWidgets.QLineEdit()
        w.setStyleSheet('border: 0px')
        w.sigChanged = w.editingFinished
        w.value = w.text
        w.setValue = w.setText
        w.sigChanging = w.textChanged
        return w
