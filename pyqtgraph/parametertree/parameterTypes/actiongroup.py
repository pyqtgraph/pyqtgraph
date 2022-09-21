from ...Qt import QtCore
from .action import ParameterControlledButton
from .basetypes import GroupParameter, GroupParameterItem
from ..ParameterItem import ParameterItem
from ...Qt import QtCore


class ActionGroupParameterItem(GroupParameterItem):
    """
    Wraps a :class:`GroupParameterItem` to manage ``bool`` parameter children. Also provides convenience buttons to
    select or clear all values at once. Note these conveniences are disabled when ``exclusive`` is *True*.
    """

    def __init__(self, param, depth):
        self.button = ParameterControlledButton()
        super().__init__(param, depth)
        self.button.clicked.connect(param.activate)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.button)

    def optsChanged(self, param, opts):
        if "button" in opts:
            buttonOpts = opts["button"] or dict(visible=False)
            self.button.updateOpts(param, buttonOpts)
            self.treeWidgetChanged()
        super().optsChanged(param, opts)


class ActionGroup(GroupParameter):
    itemClass = ActionGroupParameterItem

    sigActivated = QtCore.Signal()

    def __init__(self, **opts):
        opts.setdefault("button", {})
        super().__init__(**opts)

    def activate(self):
        self.sigActivated.emit()

    def setButtonOpts(self, **opts):
        """
        Update individual button options without replacing the entire
        button definition.
        """
        buttonOpts = self.opts.get("button", {}).copy()
        buttonOpts.update(opts)
        self.setOpts(button=buttonOpts)
