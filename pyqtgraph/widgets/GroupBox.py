from ..Qt import QtCore, QtGui, QtWidgets
from .PathButton import PathButton

__all__ = ['GroupBox']

class GroupBox(QtWidgets.QGroupBox):
    """Subclass of QGroupBox that implements collapse handle.
    """
    sigCollapseChanged = QtCore.Signal(object)
    
    def __init__(self, *args):
        QtWidgets.QGroupBox.__init__(self, *args)
        
        self._collapsed = False
        # We modify the size policy when the group box is collapsed, so 
        # keep track of the last requested policy:
        self._lastSizePlocy = self.sizePolicy()
        
        self.closePath = QtGui.QPainterPath()
        self.closePath.moveTo(0, -1)
        self.closePath.lineTo(0, 1)
        self.closePath.lineTo(1, 0)
        self.closePath.lineTo(0, -1)
        
        self.openPath = QtGui.QPainterPath()
        self.openPath.moveTo(-1, 0)
        self.openPath.lineTo(1, 0)
        self.openPath.lineTo(0, 1)
        self.openPath.lineTo(-1, 0)
        
        self.collapseBtn = PathButton(path=self.openPath, size=(12, 12), margin=0)
        self.collapseBtn.setStyleSheet("""
            border: none;
        """)
        self.collapseBtn.setPen('k')
        self.collapseBtn.setBrush('w')
        self.collapseBtn.setParent(self)
        self.collapseBtn.move(3, 3)
        self.collapseBtn.setFlat(True)
        
        self.collapseBtn.clicked.connect(self.toggleCollapsed)

        if len(args) > 0 and isinstance(args[0], str):
            self.setTitle(args[0])
        
    def toggleCollapsed(self):
        self.setCollapsed(not self._collapsed)

    def collapsed(self):
        return self._collapsed
    
    def setCollapsed(self, c):
        if c == self._collapsed:
            return
        
        if c is True:
            self.collapseBtn.setPath(self.closePath)
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred, closing=True)
        elif c is False:
            self.collapseBtn.setPath(self.openPath)
            self.setSizePolicy(self._lastSizePolicy)
        else:
            raise TypeError("Invalid argument %r; must be bool." % c)
        
        for ch in self.children():
            if isinstance(ch, QtWidgets.QWidget) and ch is not self.collapseBtn:
                ch.setVisible(not c)
        
        self._collapsed = c
        self.sigCollapseChanged.emit(c)

    def setSizePolicy(self, *args, **kwds):
        QtWidgets.QGroupBox.setSizePolicy(self, *args)
        if kwds.pop('closing', False) is True:
            self._lastSizePolicy = self.sizePolicy()

    def setHorizontalPolicy(self, *args):
        QtWidgets.QGroupBox.setHorizontalPolicy(self, *args)
        self._lastSizePolicy = self.sizePolicy()

    def setVerticalPolicy(self, *args):
        QtWidgets.QGroupBox.setVerticalPolicy(self, *args)
        self._lastSizePolicy = self.sizePolicy()

    def setTitle(self, title):
        # Leave room for button
        QtWidgets.QGroupBox.setTitle(self, "   " + title)
        
    def widgetGroupInterface(self):
        return (self.sigCollapseChanged, 
                GroupBox.collapsed, 
                GroupBox.setCollapsed, 
                True)
