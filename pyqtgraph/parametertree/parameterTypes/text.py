from .basetypes import WidgetParameterItem
from ..Parameter import Parameter
from ...Qt import QtWidgets, QtCore


class TextParameterItem(WidgetParameterItem):
    """ParameterItem displaying a QTextEdit widget."""

    def makeWidget(self):
        self.hideWidget = False
        self.asSubItem = True
        self.textBox = w = QtWidgets.QTextEdit()
        w.sizeHint = lambda: QtCore.QSize(300, 100)
        w.value = w.toPlainText
        w.setValue = w.setPlainText
        w.sigChanged = w.textChanged
        return w


class TextParameter(Parameter):
    """Editable string, displayed as large text box in the tree."""
    itemClass = TextParameterItem
