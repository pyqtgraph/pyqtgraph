__all__ = ["DockDrop"]

from ..Qt import QtCore, QtGui, QtWidgets


class DockDrop:
    """Provides dock-dropping methods"""
    def __init__(self, dndWidget):
        self.dndWidget = dndWidget
        self.allowedAreas = {'center', 'right', 'left', 'top', 'bottom'}
        self.dndWidget.setAcceptDrops(True)
        self.dropArea = None
        self.overlay = DropAreaOverlay(dndWidget)
        self.overlay.raise_()

    def addAllowedArea(self, area):
        self.allowedAreas.update(area)

    def removeAllowedArea(self, area):
        self.allowedAreas.discard(area)

    def resizeOverlay(self, size):
        self.overlay.resize(size)
        
    def raiseOverlay(self):
        self.overlay.raise_()
    
    def dragEnterEvent(self, ev):
        src = ev.source()
        if hasattr(src, 'implements') and src.implements('dock'):
            #print "drag enter accept"
            ev.accept()
        else:
            #print "drag enter ignore"
            ev.ignore()
        
    def dragMoveEvent(self, ev):
        #print "drag move"
        # QDragMoveEvent inherits QDropEvent which provides posF()
        # PyQt6 provides only position()
        width, height = self.dndWidget.width(), self.dndWidget.height()
        posF = ev.posF() if hasattr(ev, 'posF') else ev.position()
        ld = posF.x()
        rd = width - ld
        td = posF.y()
        bd = height - td
        
        mn = min(ld, rd, td, bd)
        if mn > 30:
            self.dropArea = "center"
        elif (ld == mn or td == mn) and mn > height/3:
            self.dropArea = "center"
        elif (rd == mn or ld == mn) and mn > width/3:
            self.dropArea = "center"
            
        elif rd == mn:
            self.dropArea = "right"
        elif ld == mn:
            self.dropArea = "left"
        elif td == mn:
            self.dropArea = "top"
        elif bd == mn:
            self.dropArea = "bottom"
            
        if ev.source() is self.dndWidget and self.dropArea == 'center':
            #print "  no self-center"
            self.dropArea = None
            ev.ignore()
        elif self.dropArea not in self.allowedAreas:
            #print "  not allowed"
            self.dropArea = None
            ev.ignore()
        else:
            #print "  ok"
            ev.accept()
        self.overlay.setDropArea(self.dropArea)
            
    def dragLeaveEvent(self, ev):
        self.dropArea = None
        self.overlay.setDropArea(self.dropArea)
    
    def dropEvent(self, ev):
        area = self.dropArea
        if area is None:
            return
        if area == 'center':
            area = 'above'
        self.dndWidget.area.moveDock(ev.source(), area, self.dndWidget)
        self.dropArea = None
        self.overlay.setDropArea(self.dropArea)

        

class DropAreaOverlay(QtWidgets.QWidget):
    """Overlay widget that draws drop areas during a drag-drop operation"""
    
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.dropArea = None
        self.hide()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
    def setDropArea(self, area):
        self.dropArea = area
        if area is None:
            self.hide()
        else:
            ## Resize overlay to just the region where drop area should be displayed.
            ## This works around a Qt bug--can't display transparent widgets over QGLWidget
            prgn = self.parent().rect()
            rgn = QtCore.QRect(prgn)
            w = min(30, int(prgn.width() / 3))
            h = min(30, int(prgn.height() / 3))
            
            if self.dropArea == 'left':
                rgn.setWidth(w)
            elif self.dropArea == 'right':
                rgn.setLeft(rgn.left() + prgn.width() - w)
            elif self.dropArea == 'top':
                rgn.setHeight(h)
            elif self.dropArea == 'bottom':
                rgn.setTop(rgn.top() + prgn.height() - h)
            elif self.dropArea == 'center':
                rgn.adjust(w, h, -w, -h)
            self.setGeometry(rgn)
            self.show()

        self.update()
    
    def paintEvent(self, ev):
        if self.dropArea is None:
            return
        p = QtGui.QPainter(self)
        rgn = self.rect()

        p.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 255, 50)))
        p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 150), 3))
        p.drawRect(rgn)
        p.end()
