from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem


class ParameterControlledButton(QtWidgets.QPushButton):
    settableAttributes = {"title", "tip", "icon", "shortcut", "enabled", "visible"}

    def __init__(self, parameter=None, parent=None):
        super().__init__(parent)
        if not parameter:
            return
        parameter.sigNameChanged.connect(self.onNameChange)
        parameter.sigOptionsChanged.connect(self.updateOpts)
        self.clicked.connect(parameter.activate)
        self.updateOpts(parameter, parameter.opts)

    def updateOpts(self, param, opts):
        # Of the attributes that can be set on a QPushButton, only the text
        # and tooltip attributes are different from standard pushbutton names
        nameMap = dict(title="text", tip="tooltip")
        # Special case: "title" could be none, in which case make it something
        # readable by the simple copy-paste logic later
        opts = opts.copy()
        if "name" in opts:
            opts.setdefault("title", opts["name"])
        if "title" in opts and opts["title"] is None:
            opts["title"] = param.title()

        for attr in self.settableAttributes.intersection(opts):
            buttonAttr = nameMap.get(attr, attr).title()
            setter = getattr(self, f"set{buttonAttr}")
            setter(opts[attr])

    def onNameChange(self, param, name):
        self.updateOpts(param, title=param.title())


class ActionParameterItem(ParameterItem):
    """ParameterItem displaying a clickable button."""
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self.layoutWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layoutWidget.setLayout(self.layout)
        self.button = ParameterControlledButton(param)
        #self.layout.addSpacing(100)
        self.layout.addWidget(self.button)
        self.layout.addStretch()
        self.titleChanged()

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return

        self.setFirstColumnSpanned(True)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def titleChanged(self):
        self.setSizeHint(0, self.button.sizeHint())


class ActionParameter(Parameter):
    """Used for displaying a button within the tree.

    ``sigActivated(self)`` is emitted when the button is clicked.
    """
    itemClass = ActionParameterItem
    sigActivated = QtCore.Signal(object)

    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)
