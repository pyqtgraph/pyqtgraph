from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter


class ChecklistParameterItem(GroupParameterItem):
    """
    Wraps a :class:`GroupParameterItem` to manage ``bool`` parameter children. Also provides convenience buttons to
    select or clear all values at once. Note these conveniences are disabled when ``exclusive`` is *True*.
    """
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

        self.metaBtns['default'] = self.makeDefaultButton()
        self.metaBtnLayout.addWidget(self.metaBtns['default'])

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.metaBtnWidget)

    def selectAllClicked(self):
        # timer stop: see explanation on param.setToDefault()
        self.param.valChangingProxy.timer.stop()
        self.param.setValue(self.param.reverse[0])

    def clearAllClicked(self):
        # timer stop: see explanation on param.setToDefault()
        self.param.valChangingProxy.timer.stop()
        self.param.setValue([])

    def insertChild(self, pos, item):
        ret = super().insertChild(pos, item)
        self.btnGrp.addButton(item.widget)
        return ret

    def addChild(self, item):
        ret = super().addChild(item)
        self.btnGrp.addButton(item.widget)
        return ret

    def takeChild(self, i):
        child = super().takeChild(i)
        self.btnGrp.removeButton(child.widget)

    def optsChanged(self, param, opts):
        super().optsChanged(param, opts)
        if 'expanded' in opts:
            for btn in self.metaBtns.values():
                btn.setVisible(opts['expanded'])
        exclusive = opts.get('exclusive', param.opts['exclusive'])
        enabled = opts.get('enabled', param.opts['enabled'])
        for name, btn in self.metaBtns.items():
            if name != 'default':
                btn.setDisabled(exclusive or (not enabled))
        self.btnGrp.setExclusive(exclusive)
        # "Limits" will force update anyway, no need to duplicate if it's present
        if 'limits' not in opts and ('enabled' in opts or 'readonly' in opts):
            self.updateDefaultBtn()

    def expandedChangedEvent(self, expanded):
        for btn in self.metaBtns.values():
            btn.setVisible(expanded)

    def valueChanged(self, param, val):
        self.updateDefaultBtn()

    def updateDefaultBtn(self):
        self.metaBtns["default"].setEnabled(
            not self.param.valueIsDefault()
            and self.param.opts["enabled"]
            and self.param.writable()
        )
        return

    makeDefaultButton = WidgetParameterItem.makeDefaultButton
    defaultClicked = WidgetParameterItem.defaultClicked

class RadioParameterItem(BoolParameterItem):
    """
    Allows radio buttons to function as booleans when `exclusive` is *True*
    """

    def __init__(self, param, depth):
        self.emitter = Emitter()
        super().__init__(param, depth)

    def makeWidget(self):
        w = QtWidgets.QRadioButton()
        w.value = w.isChecked
        # Since these are only used during exclusive operations, only fire a signal when "True"
        # to avoid a double-fire
        w.setValue = w.setChecked
        w.sigChanged = self.emitter.sigChanged
        w.toggled.connect(self.maybeSigChanged)
        self.hideWidget = False
        return w

    def maybeSigChanged(self, val):
        """
        Make sure to only activate on a "true" value, since an exclusive button group fires once to deactivate
        the old option and once to activate the new selection
        """
        if not val:
            return
        self.emitter.sigChanged.emit(self, val)


# Proxy around radio/bool type so the correct item class gets instantiated
class BoolOrRadioParameter(SimpleParameter):

    def __init__(self, **kargs):
        if kargs.get('type') == 'bool':
            self.itemClass = BoolParameterItem
        else:
            self.itemClass = RadioParameterItem
        super().__init__(**kargs)

class ChecklistParameter(GroupParameter):
    """
    Can be set just like a :class:`ListParameter`, but allows for multiple values to be selected simultaneously.

    ============== ========================================================
    **Options**
    exclusive      When *False*, any number of options can be selected. The resulting ``value()`` is a list of
                   all checked values. When *True*, it behaves like a ``list`` type -- only one value can be selected.
                   If no values are selected and ``exclusive`` is set to *True*, the first available limit is selected.
                   The return value of an ``exclusive`` checklist is a single value rather than a list with one element.
    delay          Controls the wait time between editing the checkboxes/radio button children and firing a "value changed"
                   signal. This allows users to edit multiple boxes at once for a single value update.
    ============== ========================================================
    """
    itemClass = ChecklistParameterItem

    def __init__(self, **opts):
        # Child options are populated through values, not explicit "children"
        if 'children' in opts:
            raise ValueError(
                "Cannot pass 'children' to ChecklistParameter. Pass a 'value' key only."
            )
        self.targetValue = None
        limits = opts.setdefault('limits', [])
        self.forward, self.reverse = ListParameter.mapping(limits)
        value = opts.setdefault('value', limits)
        opts.setdefault('exclusive', False)
        super().__init__(**opts)
        # Force 'exclusive' to trigger by making sure value is not the same
        self.sigLimitsChanged.connect(self.updateLimits)
        self.sigOptionsChanged.connect(self.optsChanged)
        if len(limits):
            # Since update signal wasn't hooked up until after parameter construction, need to fire manually
            self.updateLimits(self, limits)
            # Also, value calculation will be incorrect until children are added, so make sure to recompute
            self.setValue(value)

        self.valChangingProxy = SignalProxy(
            self.sigValueChanging,
            delay=opts.get('delay', 1.0),
            slot=self._finishChildChanges,
            threadSafe=False,
        )

    def childrenValue(self):
        vals = [self.forward[p.name()] for p in self.children() if p.value()]
        exclusive = self.opts['exclusive']
        if not vals and exclusive:
            return None
        elif exclusive:
            return vals[0]
        else:
            return vals

    def _onChildChanging(self, child, value):
        # When exclusive, ensure only this value is True
        if self.opts['exclusive'] and value:
            value = self.forward[child.name()]
        else:
            value = self.childrenValue()
        self.sigValueChanging.emit(self, value)

    def updateLimits(self, _param, limits):
        oldOpts = self.names
        val = self.opts['value']
        # Make sure adding and removing children don't cause tree state changes
        self.blockTreeChangeSignal()
        self.clearChildren()
        self.forward, self.reverse = ListParameter.mapping(limits)
        if self.opts.get('exclusive'):
            typ = 'radio'
        else:
            typ = 'bool'
        for chName in self.forward:
            # Recycle old values if they match the new limits
            newVal = bool(oldOpts.get(chName, False))
            child = BoolOrRadioParameter(type=typ, name=chName, value=newVal, default=None)
            self.addChild(child)
            # Prevent child from broadcasting tree state changes, since this is handled by self
            child.blockTreeChangeSignal()
            child.sigValueChanged.connect(self._onChildChanging)
        # Purge child changes before unblocking
        self.treeStateChanges.clear()
        self.unblockTreeChangeSignal()
        self.setValue(val)

    def _finishChildChanges(self, paramAndValue):
        param, value = paramAndValue
        # Interpret value, fire sigValueChanged
        return self.setValue(value)

    def optsChanged(self, param, opts):
        if 'exclusive' in opts:
            # Force set value to ensure updates
            # self.opts['value'] = self._VALUE_UNSET
            self.updateLimits(None, self.opts.get('limits', []))
        if 'delay' in opts:
            self.valChangingProxy.setDelay(opts['delay'])

    def setValue(self, value, blockSignal=None):
        self.targetValue = value
        if not isinstance(value, list):
            value = [value]
        names, values = self._intersectionWithLimits(value)
        valueToSet = values

        if self.opts['exclusive']:
            if len(self.forward):
                # Exclusive means at least one entry must exist, grab from limits
                # if they exist
                names.append(self.reverse[1][0])
            if len(names) > 1:
                names = names[:1]
            if not len(names):
                valueToSet = None
            else:
                valueToSet = self.forward[names[0]]

        for chParam in self:
            checked = chParam.name() in names
            # Will emit at the end, so no problem discarding existing changes
            chParam.setValue(checked, self._onChildChanging)
        super().setValue(valueToSet, blockSignal)

    def _intersectionWithLimits(self, values: list):
        """
        Returns the (names, values) from limits that intersect with ``values``.
        """
        allowedNames = []
        allowedValues = []
        # Could be replaced by "value in self.reverse[0]" and "reverse[0].index",
        # but this allows for using pg.eq to cover more diverse value options
        for val in values:
            for limitName, limitValue in zip(*self.reverse):
                if fn.eq(limitValue, val):
                    allowedNames.append(limitName)
                    allowedValues.append(val)
                    break
        return allowedNames, allowedValues

    def setToDefault(self):
        # Since changing values are covered by a proxy, this method must be overridden
        # to flush changes. Otherwise, setting to default while waiting for changes
        # to finalize will override the request to take default values
        self.valChangingProxy.timer.stop()
        super().setToDefault()

    def saveState(self, filter=None):
        # Unlike the normal GroupParameter, child states shouldn't be separately
        # preserved
        state = super().saveState(filter)
        state.pop("children", None)
        return state

    def restoreState(
        self,
        state,
        recursive=True,
        addChildren=True,
        removeChildren=True,
        blockSignals=True
    ):
        # Child management shouldn't happen through state
        return super().restoreState(
            state,
            recursive,
            addChildren=False,
            removeChildren=False,
            blockSignals=blockSignals
        )
