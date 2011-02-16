# -*- coding: utf-8 -*-
from PyQt4 import QtGui, QtCore
"""For circumventing PyQt's lack of multiple inheritance (just until PySide becomes stable)"""


class Obj(QtCore.QObject):
    def event(self, ev):
        self.emit(QtCore.SIGNAL('event'), ev)
        return QtCore.QObject.event(self, ev)

class QObjectWorkaround:
    def __init__(self):
        self._qObj_ = Obj()
        self.connect(QtCore.SIGNAL('event'), self.event)
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
    def setProperty(self, prop, val):
        return self._qObj_.setProperty(prop, val)
    def property(self, prop):
        return self._qObj_.property(prop)
    def event(self, ev):
        pass
        
#class QGraphicsObject(QtGui.QGraphicsItem, QObjectWorkaround):
    #def __init__(self, *args):
        #QtGui.QGraphicsItem.__init__(self, *args)
        #QObjectWorkaround.__init__(self)

class QGraphicsObject(QtGui.QGraphicsWidget):
    def shape(self):
        return QtGui.QGraphicsItem.shape(self)
    
#QGraphicsObject = QtGui.QGraphicsObject