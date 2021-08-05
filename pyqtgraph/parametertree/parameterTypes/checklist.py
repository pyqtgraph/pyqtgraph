from .basetypes import GroupParameterItem, GroupParameter
from .. import ParameterItem
from ...Qt import QtWidgets

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

  def treeWidgetChanged(self):
    ParameterItem.treeWidgetChanged(self)
    tw = self.treeWidget()
    if tw is None:
      return
    tw.setItemWidget(self, 1, self.metaBtnWidget)

  def selectAllClicked(self):
    self.param.setValue(list(self.param.names))

  def clearAllClicked(self):
    self.param.setValue([])

  def insertChild(self, pos, item):
    ret = super().insertChild(pos, item)
    self.btnGrp.addButton(item.widget)

  def takeChild(self, i):
    child = super().takeChild(i)
    self.btnGrp.removeButton(child.widget)

  def optsChanged(self, param, opts):
    if 'exclusive' in opts:
      excl = opts['exclusive']
      self.btnGrp.setExclusive(excl)
      for btn in self.metaBtns.values():
        btn.setDisabled(excl)
    if 'enabled' in opts:
      for btn in self.metaBtns.values():
        btn.setEnabled(opts['expanded'])

  def expandedChangedEvent(self, expanded):
    for btn in self.metaBtns.values():
      btn.setVisible(expanded)

class ChecklistParameter(GroupParameter):
  itemClass = ChecklistParameterItem

  def __init__(self, **opts):
    # Value setting before init causes problems since limits aren't set by that point.
    # Avoid by taking value out of consideration until later
    opts.setdefault('limits', [])
    opts.setdefault('value', opts['limits'])
    super().__init__(name=opts.pop('name'), exclusive=opts.pop('exclusive', False), value=opts.pop('value'))
    self.sigLimitsChanged.connect(self.updateLimits)
    self.sigOptionsChanged.connect(self.optsChanged)
    self.setLimits(opts.pop('limits', []))
    self.setOpts(**opts)

  def updateLimits(self, _param, limits):
    oldOpts = self.names
    val = self.opts['value']
    # Make sure adding and removing children don't cause tree state changes
    self.blockTreeChangeSignal()
    self.clearChildren()
    exclusive = self.opts['exclusive']
    for chOpts in limits:
      if isinstance(chOpts, str):
        # Recycle old values if they match the new limits
        newVal = oldOpts.get(chOpts, False)
        chOpts = dict(name=chOpts, value=newVal)
      child = self.create(type='bool', **chOpts)
      # Prevent child from broadcasting tree state changes
      self.addChild(child)
      child.blockTreeChangeSignal()
      child.sigValueChanged.connect(self._onSubParamChange)
    # Purge child changes before unblocking
    self.treeStateChanges.clear()
    self.unblockTreeChangeSignal()
    if exclusive and limits:
      setVal = val[-1] if val else limits[-1]
    else:
      setVal = val
    if setVal is not None:
      self.setValue(setVal)

  def _onSubParamChange(self, param, value):
    # Old value will become false, new value becomes true. Only fire on new value = true for exclusive
    if not value and self.opts['exclusive']:
      return
    elif self.opts['exclusive']:
      return self.setValue(param.name())
    # Interpret value, fire sigValueChanged
    return self.setValue(self.value())

  def optsChanged(self, param, opts):
    if 'exclusive' in opts:
      self.updateLimits(param, self.opts.get('limits', []))

  def value(self):
    vals = [p.name() for p in self.children() if p.value()]
    exclusive = self.opts['exclusive']
    if not vals and exclusive:
      return None
    elif exclusive:
      return vals[0]
    else:
      return vals

  def setValue(self, value, blockSignal=None):
    exclusive = self.opts['exclusive']
    # Will emit at the end, so no problem discarding existing changes
    cmpVal = value if isinstance(value, list) else [value]
    for chParam in self:
      checked = chParam.name() in cmpVal
      chParam.setValue(checked, self._onSubParamChange)
    super().setValue(value, blockSignal)
