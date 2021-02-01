from ..Qt import QtCore, QtGui
from ..python2_3 import asUnicode

class CmdInput(QtGui.QLineEdit):
    
    sigExecuteCmd = QtCore.Signal(object)
    
    def __init__(self, parent):
        QtGui.QLineEdit.__init__(self, parent)
        self.history = [""]
        self.ptr = 0
    
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Up:
            if self.ptr < len(self.history) - 1:
                self.setHistory(self.ptr+1)
                ev.accept()
                return
        elif ev.key() ==  QtCore.Qt.Key_Down:
            if self.ptr > 0:
                self.setHistory(self.ptr-1)
                ev.accept()
                return
        elif ev.key() == QtCore.Qt.Key_Return:
            self.execCmd()
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
