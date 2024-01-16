import warnings

from qtpy import QtCore, QtGui, QtWidgets

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
        
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        #p.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 200)))
        #p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 100)))
        #p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        #p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        
        if self.orientation == 'vertical':
            p.rotate(-90)
            rgn = QtCore.QRect(-self.height(), 0, self.height(), self.width())
        else:
            rgn = self.contentsRect()
        align = self.alignment()
        #align  = QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hint = p.drawText(rgn, align, self.text())
        p.end()
        size: QtCore.QSize = self.size()
        if self.orientation == 'vertical':
            self.setMaximumWidth(self.hint.height())
            self.setMinimumWidth(0)
            self.setMaximumHeight(16777215)
            if self.forceWidth:
                self.setMinimumHeight(self.hint.width())
            else:
                self.setMinimumHeight(0)
            size = QtCore.QSize(self.hint.height(), size.height())
        else:
            self.setMaximumHeight(self.hint.height()+5)
            self.setMinimumHeight(0)
            self.setMaximumWidth(16777215)
            if self.forceWidth:
                self.setMinimumWidth(self.hint.width())
            else:
                self.setMinimumWidth(0)
            size = QtCore.QSize(size.width(), self.hint.height()+5)
        #self.resize(size)

    def sizeHint(self):
        if self.orientation == 'vertical':
            if hasattr(self, 'hint'):
                return QtCore.QSize(self.hint.height(), self.hint.width())
            else:
                return QtCore.QSize(19, 50)
        else:
            if hasattr(self, 'hint'):
                return QtCore.QSize(self.hint.width(), self.hint.height())
            else:
                return QtCore.QSize(50, 19)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet(qdarkstyle.DarkPalette))

    label = QtWidgets.QLabel('mysuperlabel')
    labelv = VerticalLabel('mysuperlabel', orientation='hor')
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
    labelv.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
    f = label.font()
    f.setPixelSize(25)
    label.setFont(f)
    labelv.setFont(f)
    label.show()
    labelv.show()
    QtWidgets.QApplication.processEvents()
    print(f'label size hint: {label.sizeHint()}')
    print(f'labelv size hint: {labelv.sizeHint()}')
    print(f'label size: {label.size()}')
    print(f'labelv size: {labelv.size()}')
    print(f'label margins: {label.contentsMargins().top()}, {label.contentsMargins().bottom()}')
    print(f'labelv margins: {labelv.contentsMargins().top()}, {labelv.contentsMargins().bottom()}')

    sys.exit(app.exec())