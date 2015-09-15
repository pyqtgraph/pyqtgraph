import sys
from ..Qt import QtGui, QtCore
from ..SignalProxy import SignalProxy
from ..pgcollections import OrderedDict
from ..python2_3 import asUnicode, basestring


class ComboBox(QtGui.QComboBox):
    """Extends QComboBox to add extra functionality.

    * Handles dict mappings -- user selects a text key, and the ComboBox indicates
      the selected value.
    * Requires item strings to be unique
    * Remembers selected value if list is cleared and subsequently repopulated
    * setItems() replaces the items in the ComboBox and blocks signals if the
      value ultimately does not change.
    """


    def __init__(self, parent=None, items=None, default=None):
        QtGui.QComboBox.__init__(self, parent)
        self.currentIndexChanged.connect(self.indexChanged)
        self._ignoreIndexChange = False

        #self.value = default
        if 'darwin' in sys.platform: ## because MacOSX can show names that are wider than the comboBox
            self.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToMinimumContentsLength)
            #self.setMinimumContentsLength(10)
        self._chosenText = None
        self._items = OrderedDict()

        if items is not None:
            self.setItems(items)
            if default is not None:
                self.setValue(default)

    def setValue(self, value):
        """Set the selected item to the first one having the given value."""
        text = None
        for k,v in self._items.items():
            if v == value:
                text = k
                break
        if text is None:
            raise ValueError(value)

        self.setText(text)

    def setText(self, text):
        """Set the selected item to the first one having the given text."""
        ind = self.findText(text)
        if ind == -1:
            raise ValueError(text)
        #self.value = value
        self.setCurrentIndex(ind)

    def value(self):
        """
        If items were given as a list of strings, then return the currently
        selected text. If items were given as a dict, then return the value
        corresponding to the currently selected key. If the combo list is empty,
        return None.
        """
        if self.count() == 0:
            return None
        text = asUnicode(self.currentText())
        return self._items[text]

    def ignoreIndexChange(func):
        # Decorator that prevents updates to self._chosenText
        def fn(self, *args, **kwds):
            prev = self._ignoreIndexChange
            self._ignoreIndexChange = True
            try:
                ret = func(self, *args, **kwds)
            finally:
                self._ignoreIndexChange = prev
            return ret
        return fn

    def blockIfUnchanged(func):
        # decorator that blocks signal emission during complex operations
        # and emits currentIndexChanged only if the value has actually
        # changed at the end.
        def fn(self, *args, **kwds):
            prevVal = self.value()
            blocked = self.signalsBlocked()
            self.blockSignals(True)
            try:
                ret = func(self, *args, **kwds)
            finally:
                self.blockSignals(blocked)

            # only emit if the value has changed
            if self.value() != prevVal:
                self.currentIndexChanged.emit(self.currentIndex())

            return ret
        return fn

    @ignoreIndexChange
    @blockIfUnchanged
    def setItems(self, items):
        """
        *items* may be a list or a dict.
        If a dict is given, then the keys are used to populate the combo box
        and the values will be used for both value() and setValue().
        """
        prevVal = self.value()

        self.blockSignals(True)
        try:
            self.clear()
            self.addItems(items)
        finally:
            self.blockSignals(False)

        # only emit if we were not able to re-set the original value
        if self.value() != prevVal:
            self.currentIndexChanged.emit(self.currentIndex())

    def items(self):
        return self.items.copy()

    def updateList(self, items):
        # for backward compatibility
        return self.setItems(items)

    def indexChanged(self, index):
        # current index has changed; need to remember new 'chosen text'
        if self._ignoreIndexChange:
            return
        self._chosenText = asUnicode(self.currentText())

    def setCurrentIndex(self, index):
        QtGui.QComboBox.setCurrentIndex(self, index)

    def itemsChanged(self):
        # try to set the value to the last one selected, if it is available.
        if self._chosenText is not None:
            try:
                self.setText(self._chosenText)
            except ValueError:
                pass

    @ignoreIndexChange
    def insertItem(self, *args):
        raise NotImplementedError()
        #QtGui.QComboBox.insertItem(self, *args)
        #self.itemsChanged()

    @ignoreIndexChange
    def insertItems(self, *args):
        raise NotImplementedError()
        #QtGui.QComboBox.insertItems(self, *args)
        #self.itemsChanged()

    @ignoreIndexChange
    def addItem(self, *args, **kwds):
        # Need to handle two different function signatures for QComboBox.addItem
        try:
            if isinstance(args[0], basestring):
                text = args[0]
                if len(args) == 2:
                    value = args[1]
                else:
                    value = kwds.get('value', text)
            else:
                text = args[1]
                if len(args) == 3:
                    value = args[2]
                else:
                    value = kwds.get('value', text)

        except IndexError:
            raise TypeError("First or second argument of addItem must be a string.")

        if text in self._items:
            raise Exception('ComboBox already has item named "%s".' % text)

        self._items[text] = value
        QtGui.QComboBox.addItem(self, *args)
        self.itemsChanged()

    def setItemValue(self, name, value):
        if name not in self._items:
            self.addItem(name, value)
        else:
            self._items[name] = value

    @ignoreIndexChange
    @blockIfUnchanged
    def addItems(self, items):
        if isinstance(items, list):
            texts = items
            items = dict([(x, x) for x in items])
        elif isinstance(items, dict):
            texts = list(items.keys())
        else:
            raise TypeError("items argument must be list or dict (got %s)." % type(items))

        for t in texts:
            if t in self._items:
                raise Exception('ComboBox already has item named "%s".' % t)


        for k,v in items.items():
            self._items[k] = v
        QtGui.QComboBox.addItems(self, list(texts))

        self.itemsChanged()

    @ignoreIndexChange
    def clear(self):
        self._items = OrderedDict()
        QtGui.QComboBox.clear(self)
        self.itemsChanged()
