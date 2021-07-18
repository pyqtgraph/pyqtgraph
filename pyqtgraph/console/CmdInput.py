from ..Qt import QtCore, QtWidgets
from ..widgets.PopupLineEditor import PopupLineEditor
from ..python2_3 import asUnicode

class CmdInput(PopupLineEditor):
    
    sigExecuteCmd = QtCore.Signal(object)
    sigCompleteRequested = QtCore.Signal(object) # Current text

    def __init__(self, parent, *args, **kwargs):
        kwargs.setdefault('validatePrefix', False)
        kwargs.setdefault('clearOnComplete', False)
        super().__init__(parent, *args, **kwargs)
        self.history = [""]
        self.ptr = 0
    
    def keyPressEvent(self, ev):
        ctrlPressed = (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if ev.key() == QtCore.Qt.Key.Key_Up:
            if self.ptr < len(self.history) - 1:
                self.setHistory(self.ptr+1)
                ev.accept()
                return
        elif ev.key() ==  QtCore.Qt.Key.Key_Down:
            if self.ptr > 0:
                self.setHistory(self.ptr-1)
                ev.accept()
                return
        elif ev.key() == QtCore.Qt.Key.Key_Return and not self.completer().popup().isVisible():
            # Completer takes precedence if present
            self.execCmd()
        elif ctrlPressed and ev.key() == QtCore.Qt.Key.Key_Space:
            self.sigCompleteRequested.emit(self.text())
            ev.accept()
            return
        else:
            super().keyPressEvent(ev)
            self.history[0] = asUnicode(self.text())
        
    def execCmd(self):
        cmd = asUnicode(self.text())
        if len(self.history) == 1 or cmd != self.history[1]:
            self.history.insert(1, cmd)
        self.history[0] = ""
        self.setHistory(0)
        self.sigExecuteCmd.emit(cmd)
        
    def setHistory(self, num):
        self.ptr = num
        self.setText(self.history[self.ptr])
