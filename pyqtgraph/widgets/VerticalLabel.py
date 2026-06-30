import warnings

from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['VerticalLabel']
#class VerticalLabel(QtWidgets.QLabel):
    #def paintEvent(self, ev):
        #p = QtGui.QPainter(self)
        #p.rotate(-90)
        #self.hint = p.drawText(QtCore.QRect(-self.height(), 0, self.height(), self.width()), QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter, self.text())
        #p.end()
        #self.setMinimumWidth(self.hint.height())
        #self.setMinimumHeight(self.hint.width())

    #def sizeHint(self):
        #if hasattr(self, 'hint'):
            #return QtCore.QSize(self.hint.height(), self.hint.width())
        #else:
            #return QtCore.QSize(16, 50)

class VerticalLabel(QtWidgets.QLabel):
    def __init__(self, text, orientation='vertical', forceWidth=True):
        QtWidgets.QLabel.__init__(self, text)
        self.forceWidth = forceWidth
        self.orientation = None
        self.setOrientation(orientation)
        
    def setOrientation(self, o):
        if self.orientation == o:
            return
        self.orientation = o
        self.update()
        self.updateGeometry()

    def _textSize(self):
        metrics = self.fontMetrics()
        return QtCore.QSize(metrics.horizontalAdvance(self.text()), metrics.height())
        
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        #p.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 200)))
        #p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 100)))
        #p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        #p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        
        contents = self.contentsRect()
        if self.orientation == 'vertical':
            p.rotate(-90)
            rgn = QtCore.QRect(
                -contents.y() - contents.height(),
                contents.x(),
                contents.height(),
                contents.width()
            )
        else:
            rgn = contents
        align = self.alignment()
        #align  = QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.drawText(rgn, align, self.text())
        self.hint = self._textSize()
        p.end()

        margins = self.contentsMargins()
        textWidth = self.hint.width()
        textHeight = self.hint.height()
        paddedWidth = textWidth + margins.left() + margins.right()
        paddedHeight = textHeight + margins.top() + margins.bottom()
        
        if self.orientation == 'vertical':
            self.setMaximumWidth(textHeight + margins.left() + margins.right())
            self.setMinimumWidth(0)
            self.setMaximumHeight(16777215)
            if self.forceWidth:
                self.setMinimumHeight(textWidth + margins.top() + margins.bottom())
            else:
                self.setMinimumHeight(0)
        else:
            self.setMaximumHeight(paddedHeight)
            self.setMinimumHeight(0)
            self.setMaximumWidth(16777215)
            if self.forceWidth:
                self.setMinimumWidth(paddedWidth)
            else:
                self.setMinimumWidth(0)

    def sizeHint(self):
        margins = self.contentsMargins()
        hint = self._textSize()
        if self.orientation == 'vertical':
            return QtCore.QSize(
                hint.height() + margins.left() + margins.right(),
                hint.width() + margins.top() + margins.bottom()
            )
        else:
            return QtCore.QSize(
                hint.width() + margins.left() + margins.right(),
                hint.height() + margins.top() + margins.bottom()
            )
