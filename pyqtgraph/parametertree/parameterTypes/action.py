from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem


class ActionParameterItem(ParameterItem):
    """ParameterItem displaying a clickable button."""
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self.layoutWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layoutWidget.setLayout(self.layout)
        self.button = QtWidgets.QPushButton()
        #self.layout.addSpacing(100)
        self.layout.addWidget(self.button)
        self.layout.addStretch()
        self.button.clicked.connect(self.buttonClicked)
        self.titleChanged()
        self.optsChanged(self.param, self.param.opts)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return

        self.setFirstColumnSpanned(True)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def titleChanged(self):
        self.button.setText(self.param.title())
        self.setSizeHint(0, self.button.sizeHint())

    def optsChanged(self, param, opts):
        ParameterItem.optsChanged(self, param, opts)

        if 'enabled' in opts:
            self.button.setEnabled(opts['enabled'])

        if 'tip' in opts:
            self.button.setToolTip(opts['tip'])

    def buttonClicked(self):
        self.param.activate()


class ActionParameter(Parameter):
    """Used for displaying a button within the tree.

    ``sigActivated(self)`` is emitted when the button is clicked.
    """
    itemClass = ActionParameterItem
    sigActivated = QtCore.Signal(object)

    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)
