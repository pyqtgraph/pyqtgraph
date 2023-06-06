from math import hypot

from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['JoystickButton']

class JoystickButton(QtWidgets.QPushButton):
    sigStateChanged = QtCore.Signal(object, object)  ## self, state
    
    def __init__(self, parent=None):
        QtWidgets.QPushButton.__init__(self, parent)
        self.radius = 200
        self.setCheckable(True)
        self.state = None
        self.setState(0, 0)
        self.setFixedWidth(50)
        self.setFixedHeight(50)
        
        
    def mousePressEvent(self, ev):
        self.setChecked(True)
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.pressPos = lpos
        ev.accept()
        
    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        dif = lpos - self.pressPos
        self.setState(dif.x(), -dif.y())
        
    def mouseReleaseEvent(self, ev):
        self.setChecked(False)
        self.setState(0,0)
        
    def wheelEvent(self, ev):
        ev.accept()
        
        
    def doubleClickEvent(self, ev):
        ev.accept()
        
    def getState(self):
        return self.state
        
    def setState(self, *xy):
        xy = list(xy)
        d = hypot(xy[0], xy[1])  # length
        nxy = [0, 0]
        for i in [0,1]:
            if xy[i] == 0:
                nxy[i] = 0
            else:
                nxy[i] = xy[i] / d
        
        if d > self.radius:
            d = self.radius
        d = (d / self.radius) ** 2
        xy = [nxy[0] * d, nxy[1] * d]
        
        w2 = self.width() / 2
        h2 = self.height() / 2
        self.spotPos = QtCore.QPoint(
            int(w2 * (1 + xy[0])),
            int(h2 * (1 - xy[1]))
        )
        self.update()
        if self.state == xy:
            return
        self.state = xy
        self.sigStateChanged.emit(self, self.state)
        
    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QtGui.QPainter(self)
        p.setBrush(QtGui.QBrush(QtGui.QColor(0,0,0)))
        p.drawEllipse(
            self.spotPos.x() - 3,
            self.spotPos.y() - 3,
            6,
            6
        )
        p.end()
        
    def resizeEvent(self, ev):
        self.setState(*self.state)
        super().resizeEvent(ev)
