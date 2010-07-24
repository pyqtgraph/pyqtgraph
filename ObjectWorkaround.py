# -*- coding: utf-8 -*-
from PyQt4 import QtGui, QtCore

class QObjectWorkaround:
    def __init__(self):
        self._qObj_ = QtCore.QObject()
    def connect(self, *args):
        if args[0] is self:
            return QtCore.QObject.connect(self._qObj_, *args[1:])
        else:
            return QtCore.QObject.connect(self._qObj_, *args)
    def disconnect(self, *args):
        return QtCore.QObject.disconnect(self._qObj_, *args)
    def emit(self, *args):
        return QtCore.QObject.emit(self._qObj_, *args)
    def blockSignals(self, b):
        return self._qObj_.blockSignals(b)
        
class QGraphicsObject(QtGui.QGraphicsItem, QObjectWorkaround):
    def __init__(self, *args):
        QtGui.QGraphicsItem.__init__(self, *args)
        QObjectWorkaround.__init__(self)
