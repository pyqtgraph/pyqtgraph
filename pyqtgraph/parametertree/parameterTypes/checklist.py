from .basetypes import GroupParameterItem, GroupParameter, WidgetParameterItem
from .list import ListParameter
from .. import ParameterItem
from ...Qt import QtWidgets
from ... import functions as fn

class ChecklistParameterItem(GroupParameterItem):
    def __init__(self, param, depth):
        self.btnGrp = QtWidgets.QButtonGroup()
        self.btnGrp.setExclusive(False)
        self._constructMetaBtns()

        super().__init__(param, depth)

    def _constructMetaBtns(self):
        self.metaBtnWidget = QtWidgets.QWidget()
        self.metaBtnLayout = lay = QtWidgets.QHBoxLayout(self.metaBtnWidget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.metaBtns = {}
        lay.addStretch(0)
        for title in 'Clear', 'Select':
            self.metaBtns[title] = btn = QtWidgets.QPushButton(f'{title} All')
            self.metaBtnLayout.addWidget(btn)
            btn.clicked.connect(getattr(self, f'{title.lower()}AllClicked'))

        self.metaBtns['default'] = WidgetParameterItem.makeDefaultButton(self)
        self.metaBtnLayout.addWidget(self.metaBtns['default'])

    def defaultClicked(self):
        self.param.setToDefault()

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.metaBtnWidget)

    def selectAllClicked(self):
        self.param.setValue(self.param.reverse[0])

    def clearAllClicked(self):
        self.param.setValue([])

    def insertChild(self, pos, item):
        ret = super().insertChild(pos, item)
        self.btnGrp.addButton(item.widget)
        return ret

    def takeChild(self, i):
        child = super().takeChild(i)
        self.btnGrp.removeButton(child.widget)
        return child

    def optsChanged(self, param, opts):
        if 'expanded' in opts:
            for btn in self.metaBtns.values():
                btn.setVisible(opts['expanded'])
        exclusive = opts.get('exclusive', param.opts['exclusive'])
        enabled = opts.get('enabled', param.opts['enabled'])
        for btn in self.metaBtns.values():
            btn.setDisabled(exclusive or (not enabled))

    def expandedChangedEvent(self, expanded):
        for btn in self.metaBtns.values():
            btn.setVisible(expanded)

class ChecklistParameter(GroupParameter):
    itemClass = ChecklistParameterItem

    # Sentinel for forcing updates by guaranteeing non-existence
    _VALUE_UNSET = object()

    def __init__(self, **opts):
        # Value setting before init causes problems since limits aren't set by that point.
        # Avoid by taking value out of consideration until later
        self.targetValue = None
        limits = opts.setdefault('limits', [])
        self.forward, self.reverse = ListParameter.mapping(limits)
        opts.setdefault('value', limits)
        opts.setdefault('exclusive', False)
        super().__init__(**opts)
        # Force 'exclusive' to trigger by making sure value is not the same
        self.sigLimitsChanged.connect(self.updateLimits)
        self.sigOptionsChanged.connect(self.optsChanged)
        if len(limits):
            # Since update signal wasn't hooked up until after parameter construction, need to fire manually
            self.updateLimits(self, limits)

    def updateLimits(self, _param, limits):
        oldOpts = self.names
        val = self.opts['value']
        # Make sure adding and removing children don't cause tree state changes
        self.blockTreeChangeSignal()
        self.clearChildren()
        self.forward, self.reverse = ListParameter.mapping(limits)
        for chName in self.forward:
            # Recycle old values if they match the new limits
            newVal = bool(oldOpts.get(chName, False))
            child = self.create(type='bool', name=chName, value=newVal, default=None)
            self.addChild(child)
            # Prevent child from broadcasting tree state changes, since this is handled by self
            child.blockTreeChangeSignal()
            child.sigValueChanged.connect(self._onSubParamChange)
        # Purge child changes before unblocking
        self.treeStateChanges.clear()
        self.unblockTreeChangeSignal()
        self.setValue(val)

    def _onSubParamChange(self, param, value):
        # Old value will become false, new value becomes true. Only fire on new value = true for exclusive
        if not value and self.opts['exclusive']:
            # Re-allow selection
            return
        elif self.opts['exclusive']:
            val = self.reverse[0][self.reverse[1].index(param.name())]
            return self.setValue(val)
        # Interpret value, fire sigValueChanged
        return self.setValue(self.value())

    def optsChanged(self, param, opts):
        if 'exclusive' in opts:
            # Force set value to ensure updates
            # self.opts['value'] = self._VALUE_UNSET
            self.setValue(self.opts['value'])

    def value(self):
        vals = [self.forward[p.name()] for p in self.children() if p.value()]
        exclusive = self.opts['exclusive']
        if not vals and exclusive:
            return None
        elif exclusive:
            return vals[0]
        else:
            return vals

    def setValue(self, value, blockSignal=None):
        self.targetValue = value
        exclusive = self.opts['exclusive']
        # Will emit at the end, so no problem discarding existing changes
        cmpVals = value if isinstance(value, list) else [value]
        for ii in range(len(cmpVals)-1, -1, -1):
            exists = any(fn.eq(cmpVals[ii], lim) for lim in self.reverse[0])
            if not exists:
                del cmpVals[ii]
        names = [self.reverse[1][self.reverse[0].index(val)] for val in cmpVals]
        if exclusive and len(names) > 1:
            names = [names[0]]
        elif exclusive and not len(names) and len(self.forward):
            # An option is required during exclusivity
            names = [self.reverse[1][0]]
        for chParam in self:
            checked = chParam.name() in names
            chParam.setValue(checked, self._onSubParamChange)
            chParam.setOpts(enabled=not (exclusive and checked))
        super().setValue(value, blockSignal)
