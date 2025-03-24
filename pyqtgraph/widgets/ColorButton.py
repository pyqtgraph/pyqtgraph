from .. import functions as functions
from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['ColorButton']

class ColorButton(QtWidgets.QPushButton):
    """
    **Bases:** QtWidgets.QPushButton
    
    Button displaying a color and allowing the user to select a new color.
    
    ====================== ============================================================
    **Signals:**
    sigColorChanging(self) emitted whenever a new color is picked in the color dialog
    sigColorChanged(self)  emitted when the selected color is accepted (user clicks OK)
    ====================== ============================================================
    """
    sigColorChanging = QtCore.Signal(object)  ## emitted whenever a new color is picked in the color dialog
    sigColorChanged = QtCore.Signal(object)   ## emitted when the selected color is accepted (user clicks OK)
    
    def __init__(self, parent=None, color=(128,128,128), padding=6):
        QtWidgets.QPushButton.__init__(self, parent)
        self.padding = (padding, padding, -padding, -padding) if isinstance(padding, (int, float)) else padding
        self.setColor(color)
        self.colorDialog = QtWidgets.QColorDialog()
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        self.colorDialog.currentColorChanged.connect(self.dialogColorChanged)
        self.colorDialog.rejected.connect(self.colorRejected)
        self.colorDialog.colorSelected.connect(self.colorSelected)
        #QtCore.QObject.connect(self.colorDialog, QtCore.SIGNAL('currentColorChanged(const QColor&)'), self.currentColorChanged)
        #QtCore.QObject.connect(self.colorDialog, QtCore.SIGNAL('rejected()'), self.currentColorRejected)
        self.clicked.connect(self.selectColor)
        self.setMinimumHeight(15)
        self.setMinimumWidth(15)
        
    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QtGui.QPainter(self)
        rect = self.rect().adjusted(*self.padding)
        ## draw white base, then texture for indicating transparency, then actual color
        p.setBrush(functions.mkBrush('w'))
        p.drawRect(rect)
        p.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.DiagCrossPattern))
        p.drawRect(rect)
        p.setBrush(functions.mkBrush(self._color))
        p.drawRect(rect)
        p.end()
    
    def setColor(self, color, finished=True):
        """Sets the button's color and emits both sigColorChanged and sigColorChanging."""
        self._color = functions.mkColor(color)
        self.update()
        if finished:
            self.sigColorChanged.emit(self)
        else:
            self.sigColorChanging.emit(self)
        
    @QtCore.Slot()
    def selectColor(self):
        self.origColor = self.color()
        self.colorDialog.setCurrentColor(self.color())
        self.colorDialog.open()
        
    @QtCore.Slot(QtGui.QColor)
    def dialogColorChanged(self, color):
        if color.isValid():
            self.setColor(color, finished=False)
            
    @QtCore.Slot()
    @QtCore.Slot(QtGui.QColor)
    def colorRejected(self):
        self.setColor(self.origColor, finished=False)
    
    @QtCore.Slot(QtGui.QColor)
    def colorSelected(self, color):
        self.setColor(self._color, finished=True)
    
    def saveState(self):
        return self._color.getRgb()
        
    def restoreState(self, state):
        self.setColor(state)
        
    def color(self, mode='qcolor'):
        color = functions.mkColor(self._color)
        if mode == 'qcolor':
            return color
        elif mode == 'byte':
            return color.getRgb()
        elif mode == 'float':
            return color.getRgbF()

    def widgetGroupInterface(self):
        return (self.sigColorChanged, ColorButton.saveState, ColorButton.restoreState)
    
