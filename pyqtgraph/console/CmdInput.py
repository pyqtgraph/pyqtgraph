# -*- coding: utf-8 -*-
from ..Qt import QtCore, QtGui

class CmdInput(QtGui.QLineEdit):
    
    sigExecuteCmd = QtCore.Signal(object)
    
    def __init__(self, parent):
        QtGui.QLineEdit.__init__(self, parent)
        self.history = [""]
        self.ptr = 0
    
    def keyPressEvent(self, ev):
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
        elif ev.key() == QtCore.Qt.Key.Key_Return:
            self.execCmd()
        else:
            super().keyPressEvent(ev)
            self.history[0] = self.text()
        
    def execCmd(self):
        cmd = self.text()
        if len(self.history) == 1 or cmd != self.history[1]:
            self.history.insert(1, cmd)
        self.history[0] = ""
        self.setHistory(0)
        self.sigExecuteCmd.emit(cmd)
        
    def setHistory(self, num):
        self.ptr = num
        self.setText(self.history[self.ptr])
