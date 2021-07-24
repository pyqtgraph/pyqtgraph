from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem


class StrParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def makeWidget(self):
        w = QtWidgets.QLineEdit()
        w.setStyleSheet('border: 0px')
        w.sigChanged = w.editingFinished
        w.value = lambda: str(w.text())
        w.setValue = lambda v: w.setText(str(v))
        w.sigChanging = w.textChanged
        return w