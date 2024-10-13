import warnings

from ...Qt import QtCore
from .action import ParameterControlledButton
from .basetypes import GroupParameter, GroupParameterItem
from ..ParameterItem import ParameterItem
from ...Qt import QtCore, QtWidgets


class ActionGroupParameterItem(GroupParameterItem):
    """
    Wraps a :class:`GroupParameterItem` to manage function parameters created by
    an interactor. Provies a button widget which mimics an ``action`` parameter.
    """

    def __init__(self, param, depth):
        self.itemWidget = QtWidgets.QWidget()
        self.button = ParameterControlledButton(parent=self.itemWidget)
        self.button.clicked.connect(param.activate)

        self.itemWidget.setLayout(layout := QtWidgets.QHBoxLayout())
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)

        super().__init__(param, depth)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.itemWidget)

    def optsChanged(self, param, opts):
        if "button" in opts:
            buttonOpts = opts["button"] or dict(visible=False)
            self.button.updateOpts(param, buttonOpts)
        super().optsChanged(param, opts)


class ActionGroupParameter(GroupParameter):
    itemClass = ActionGroupParameterItem

    sigActivated = QtCore.Signal(object)

    def __init__(self, **opts):
        opts.setdefault("button", {})
        super().__init__(**opts)

    @QtCore.Slot()
    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)

    def setButtonOpts(self, **opts):
        """
        Update individual button options without replacing the entire
        button definition.
        """
        buttonOpts = self.opts.get("button", {}).copy()
        buttonOpts.update(opts)
        self.setOpts(button=buttonOpts)


class ActionGroup(ActionGroupParameter):
    sigActivated = QtCore.Signal()

    def __init__(self, **opts):
        warnings.warn(
            "`ActionGroup` is deprecated and will be removed in the first release after "
            "January 2023. Use `ActionGroupParameter` instead. See "
            "https://github.com/pyqtgraph/pyqtgraph/pull/2505 for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**opts)

    def activate(self):
        self.sigActivated.emit()
