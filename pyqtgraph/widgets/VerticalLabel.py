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

    def _paddedTextSize(self, textSize=None):
        if textSize is None:
            textSize = self._textSize()

        margins = self.contentsMargins()
        if self.orientation == 'vertical':
            return QtCore.QSize(
                textSize.height() + margins.left() + margins.right(),
                textSize.width() + margins.top() + margins.bottom()
            )
        return QtCore.QSize(
            textSize.width() + margins.left() + margins.right(),
            textSize.height() + margins.top() + margins.bottom()
        )
        
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

        paddedSize = self._paddedTextSize(self.hint)
        
        if self.orientation == 'vertical':
            self.setMaximumWidth(paddedSize.width())
            self.setMinimumWidth(0)
            self.setMaximumHeight(16777215)
            if self.forceWidth:
                self.setMinimumHeight(paddedSize.height())
            else:
                self.setMinimumHeight(0)
        else:
            self.setMaximumHeight(paddedSize.height())
            self.setMinimumHeight(0)
            self.setMaximumWidth(16777215)
            if self.forceWidth:
                self.setMinimumWidth(paddedSize.width())
            else:
                self.setMinimumWidth(0)

    def sizeHint(self):
        return self._paddedTextSize()
