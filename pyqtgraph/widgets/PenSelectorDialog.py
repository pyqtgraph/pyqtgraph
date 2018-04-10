from pyqtgraph.Qt import QtCore, QtGui
import PenSelectorDialogbox

class PenSelectorDialog(QtGui.QDialog):
    penChanged = QtCore.pyqtSignal(object)

    def __init__(self, initialPen = None, parent = None):
        QtGui.QDialog.__init__(self, parent)
        self.ui = PenSelectorDialogbox.Ui_Dialog()
        self.ui.setupUi(self)

        self.lineStyles = [QtCore.Qt.SolidLine,QtCore.Qt.DashLine,QtCore.Qt.DotLine,QtCore.Qt.DashDotLine,QtCore.Qt.DashDotDotLine]#,QtCore.Qt.CustomDashLine]
        comboboxBackgroundColor = self.ui.comboBoxPenStyle.palette().base().color()
        comboboxSize = self.ui.comboBoxPenStyle.size()
        w,h = comboboxSize.width(),comboboxSize.height()
        self.ui.comboBoxPenStyle.setIconSize(comboboxSize)
        #create style images:
        for s in self.lineStyles:
            p = QtGui.QPixmap(comboboxSize)
            p.fill(comboboxBackgroundColor)
            painter = QtGui.QPainter(p)
            painter.setPen(QtGui.QPen(QtCore.Qt.black,2,s))
            painter.drawLine(0,h/2,w,h/2)
            painter.end()
            self.ui.comboBoxPenStyle.addItem(QtGui.QIcon(p),QtCore.QString(),QtCore.QVariant(s))

        #create cap and end options
        self.capStyles = [QtCore.Qt.SquareCap,QtCore.Qt.FlatCap,QtCore.Qt.RoundCap]
        self.joinStyles = [QtCore.Qt.BevelJoin,QtCore.Qt.MiterJoin,QtCore.Qt.RoundJoin]
        self.capStylesNames = ["Square cap","Flat cap","Round cap"]
        self.joinStylesNames = ["Bevel join","Miter join","Round join"]

        for s,n in zip(self.capStyles,self.capStylesNames):
            self.ui.comboBoxPenCapStyle.addItem(n,QtCore.QVariant(s))

        for s,n in zip(self.joinStyles,self.joinStylesNames):
            self.ui.comboBoxPenJoinStyle.addItem(n,QtCore.QVariant(s))


        self.ui.comboBoxPenStyle.currentIndexChanged.connect(self.updatePen)
        self.ui.comboBoxPenCapStyle.currentIndexChanged.connect(self.updatePen)
        self.ui.comboBoxPenJoinStyle.currentIndexChanged.connect(self.updatePen)
        self.ui.pushButtonPenColor.sigColorChanged.connect(self.updatePen)
        self.ui.doubleSpinBoxPenWidth.valueChanged.connect(self.updatePen)
        self.pen = None

        if initialPen is not None:
            self.blockSignals(True)
            self.ui.comboBoxPenStyle.setCurrentIndex(self.ui.comboBoxPenStyle.findData(initialPen.style()))
            self.ui.comboBoxPenCapStyle.setCurrentIndex(self.ui.comboBoxPenCapStyle.findData(initialPen.capStyle()))
            self.ui.comboBoxPenJoinStyle.setCurrentIndex(self.ui.comboBoxPenJoinStyle.findData(initialPen.joinStyle()))
            self.ui.pushButtonPenColor.setColor(initialPen.color())
            self.ui.doubleSpinBoxPenWidth.setValue(initialPen.width())
            self.blockSignals(False)

        self.updatePen(0)

    def updatePen(self, dummy):
        penStyle = self.ui.comboBoxPenStyle.itemData(self.ui.comboBoxPenStyle.currentIndex()).toInt()[0]
        penCapStyle = self.ui.comboBoxPenCapStyle.itemData(self.ui.comboBoxPenCapStyle.currentIndex()).toInt()[0]
        penJoinStyle = self.ui.comboBoxPenJoinStyle.itemData(self.ui.comboBoxPenJoinStyle.currentIndex()).toInt()[0]
        penWidth = self.ui.doubleSpinBoxPenWidth.value()
        penColor = self.ui.pushButtonPenColor.color()
        self.pen= QtGui.QPen(penColor,penWidth,penStyle,penCapStyle,penJoinStyle)

        displaySize = self.ui.labelPenPreview.size()
        labelBackgroundColor = self.ui.labelPenPreview.palette().background().color()
        w,h = displaySize.width(),displaySize.height()
        
        #draw a squigly line to show what the pen looks like.
        path = QtGui.QPainterPath()
        path.moveTo(w*.2,h*.2)
        path.lineTo(w*.8,h*.2)
        path.lineTo(w*.2,h*.5)
        path.cubicTo(w*.1,h*1,w*.5,h*.25,w*.8,h*.8)


        p = QtGui.QPixmap(displaySize)
        p.fill(labelBackgroundColor)
        painter = QtGui.QPainter(p)
        painter.setPen(self.pen)
        painter.drawPath(path)
        painter.end()
        self.ui.labelPenPreview.setPixmap(p)
        self.penChanged.emit(self.pen)

