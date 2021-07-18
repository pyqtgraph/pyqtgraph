from ..Qt import QtWidgets, QtCore, QtGui


class StringListValidator(QtGui.QValidator):

    def __init__(self, parent=None,strList=None, model=None, validateCase=False):
        super().__init__(parent)
        self.validateCase = validateCase
        self.strList = strList
        self.model = model

    def validate(self, input_, pos):
        if self.model:
            self.strList = [self.model.index(ii, 0).data() for ii in range(self.model.rowCount())]
        strList = cmpStrList = [input_] if self.strList is None else self.strList
        cmpInput = input_
        if not self.validateCase:
            cmpInput = input_.lower()
            cmpStrList = [s.lower() for s in strList]

        try:
            matchIdx = cmpStrList.index(cmpInput)
            input_ = strList[matchIdx]
            state = self.State.Acceptable
        except ValueError:
            if any(cmpInput in str_ for str_ in cmpStrList):
                state = self.State.Intermediate
            else:
                state = self.State.Invalid
        return state, input_, pos

class PopupLineEditor(QtWidgets.QLineEdit):
    def __init__(self, parent=None,
                 model=None,
                 placeholderText='Press Tab or type...',
                 clearOnComplete=True,
                 forceMatch=True,
                 validatePrefix=True,
                 validateCase=False):
        super().__init__(parent)
        self.setPlaceholderText(placeholderText)
        self.clearOnComplete = clearOnComplete
        self.forceMatch = forceMatch
        self.validateCase = validateCase
        self.model: QtCore.QAbstractListModel = QtCore.QStringListModel()
        if model is None:
            model = self.model

        if validatePrefix:
            self.vdator = StringListValidator(parent=self, validateCase=validateCase)
            self.setValidator(self.vdator)
        else:
            self.vdator = None

        self.setModel(model)

    def setModel(self, model):
        completer = QtWidgets.QCompleter(model, self)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionRole(QtCore.Qt.ItemDataRole.DisplayRole)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        if self.clearOnComplete:
            def clearLater():
                QtCore.QTimer.singleShot(0, self.clear)
            completer.activated.connect(clearLater)

        self.textChanged.connect(self.resetCompleterPrefix)

        self.setCompleter(completer)
        self.model = model
        if self.forceMatch and self.vdator is not None:
            self.vdator.model = model

    def _chooseNextCompletion(self, incAmt=1):
        completer = self.completer()
        popup = completer.popup()
        if popup.isVisible() and popup.currentIndex().isValid():
            nextIdx = (completer.currentRow()+incAmt)%completer.completionCount()
            completer.setCurrentRow(nextIdx)
        else:
            completer.complete()
        popup.show()
        popup.setCurrentIndex(completer.currentIndex())
        popup.setFocus()

    def event(self, ev):
        if ev.type() != ev.Type.KeyPress:
            return super().event(ev)

        ev: QtGui.QKeyEvent
        key = ev.key()
        if key == QtCore.Qt.Key.Key_Tab:
            incAmt = 1
        elif key == QtCore.Qt.Key.Key_Backtab:
            incAmt = -1
        else:
            return super().event(ev)
        self._chooseNextCompletion(incAmt)
        return True

    def focusOutEvent(self, ev):
        reason = ev.reason()
        if reason in [QtCore.Qt.FocusReason.TabFocusReason, QtCore.Qt.FocusReason.BacktabFocusReason,
                      QtCore.Qt.FocusReason.OtherFocusReason]:
            # Simulate tabbing through completer options instead of losing focus
            self.setFocus()
            completer = self.completer()
            if completer is None:
                return
            incAmt = 1 if reason == QtCore.Qt.FocusReason.TabFocusReason else -1

            self._chooseNextCompletion(incAmt)
            ev.accept()
            return
        else:
            super().focusOutEvent(ev)

    def clear(self):
        super().clear()

    def resetCompleterPrefix(self):
        if self.text() == '':
            self.completer().setCompletionPrefix('')
