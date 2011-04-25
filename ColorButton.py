# -*- coding: utf-8 -*-
from PyQt4 import QtGui, QtCore
if not hasattr(QtCore, 'Signal'):
    QtCore.Signal = QtCore.pyqtSignal
import functions

class ColorButton(QtGui.QPushButton):
    
    sigColorChanging = QtCore.Signal(object)  ## emitted whenever a new color is picked in the color dialog
    sigColorChanged = QtCore.Signal(object)   ## emitted when the selected color is accepted (user clicks OK)
    
    def __init__(self, parent=None, color=(128,128,128)):
        QtGui.QPushButton.__init__(self, parent)
        self.setColor(color)
        self.colorDialog = QtGui.QColorDialog()
        self.colorDialog.setOption(QtGui.QColorDialog.ShowAlphaChannel, True)
        self.colorDialog.setOption(QtGui.QColorDialog.DontUseNativeDialog, True)
        self.colorDialog.currentColorChanged.connect(self.dialogColorChanged)
        self.colorDialog.rejected.connect(self.colorRejected)
        self.colorDialog.colorSelected.connect(self.colorSelected)
        #QtCore.QObject.connect(self.colorDialog, QtCore.SIGNAL('currentColorChanged(const QColor&)'), self.currentColorChanged)
        #QtCore.QObject.connect(self.colorDialog, QtCore.SIGNAL('rejected()'), self.currentColorRejected)
        self.clicked.connect(self.selectColor)
        self.setMinimumHeight(15)
        self.setMinimumWidth(15)
        
    def paintEvent(self, ev):
        QtGui.QPushButton.paintEvent(self, ev)
        p = QtGui.QPainter(self)
        p.setBrush(functions.mkBrush(self._color))
        p.drawRect(self.rect().adjusted(5, 5, -5, -5))
        p.end()
    
    def setColor(self, color, finished=True):
        self._color = functions.mkColor(color)
        if finished:
            self.sigColorChanged.emit(self)
        else:
            self.sigColorChanging.emit(self)
        self.update()
        
    def selectColor(self):
        self.origColor = self.color()
        self.colorDialog.setCurrentColor(self.color())
        self.colorDialog.open()
        
    def dialogColorChanged(self, color):
        if color.isValid():
            self.setColor(color, finished=False)
            
    def colorRejected(self):
        self.setColor(self.origColor, finished=False)
    
    def colorSelected(self, color):
        self.setColor(self._color, finished=True)
    
    def saveState(self):
        return functions.colorTuple(self._color)
        
    def restoreState(self, state):
        self.setColor(state)
        
    def color(self):
        return functions.mkColor(self._color)

    def widgetGroupInterface(self):
        return (self.sigColorChanged, ColorButton.saveState, ColorButton.restoreState)
    
if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    btn = ColorButton()
    win.setCentralWidget(btn)
    win.show()
    
    def change(btn):
        print "change", btn.color()
    def done(btn):
        print "done", btn.color()
        
    btn.sigColorChanging.connect(change)
    btn.sigColorChanged.connect(done)
    