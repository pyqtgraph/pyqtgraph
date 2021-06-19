from ..Qt import QtCore, QtGui
from .PenSelectorDialogbox import PenSelectorDialogbox
from ..functions import mkColor

class PenSelectorDialog(QtGui.QDialog):
    penChanged = QtCore.Signal(object)

    def __init__(self, initialPen = None, parent = None):
        QtGui.QDialog.__init__(self, parent)
        self.ui = PenSelectorDialogbox()
        self.ui.setupUi(self)
        # Combo boxes don't like enum values, so they get converted to int. But, PySide/Qt6 doesn't accept int
        # pen constructor values, so map the int to enum values. PySide/Qt6 Enums let you iterate over them, but
        # PyQt5 doesn't, so construct manually for compatibility
        ps = QtCore.Qt.PenStyle
        lineStyles = [ps.SolidLine,ps.DashLine,ps.DotLine,ps.DashDotLine,ps.DashDotDotLine]#,ps.CustomDashLine]
        self.lineStyles = {int(style): style for style in lineStyles}
        comboboxBackgroundColor = self.ui.comboBoxPenStyle.palette().base().color()
        comboboxSize = self.ui.comboBoxPenStyle.size()
        w,h = comboboxSize.width(),comboboxSize.height()
        self.ui.comboBoxPenStyle.setIconSize(comboboxSize)
        #create style images:
        for s in self.lineStyles.values():
            p = QtGui.QPixmap(comboboxSize)
            p.fill(comboboxBackgroundColor)
            painter = QtGui.QPainter(p)
            painter.setPen(QtGui.QPen(mkColor('k'),2,s))
            painter.drawLine(0,h/2,w,h/2)
            painter.end()
            self.ui.comboBoxPenStyle.addItem(QtGui.QIcon(p), '', s)

        #create cap and end options
        cs = QtCore.Qt.PenCapStyle
        js = QtCore.Qt.PenJoinStyle
        capStyles = [cs.SquareCap,cs.FlatCap,cs.RoundCap]
        self.capStyles = {int(style): style for style in capStyles}
        joinStyles = [js.BevelJoin,js.MiterJoin,js.RoundJoin]
        self.joinStyles = {int(style): style for style in joinStyles}
        self.capStylesNames = ["Square cap","Flat cap","Round cap"]
        self.joinStylesNames = ["Bevel join","Miter join","Round join"]

        for s,n in zip(self.capStyles,self.capStylesNames):
            self.ui.comboBoxPenCapStyle.addItem(n,s)

        for s,n in zip(self.joinStyles,self.joinStylesNames):
            self.ui.comboBoxPenJoinStyle.addItem(n,s)


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
            self.ui.doubleSpinBoxPenWidth.setValue(initialPen.widthF())
            self.blockSignals(False)

        self.updatePen(0)

    def updatePen(self, dummy):
        cboxArgs = []
        cboxes = [self.ui.comboBoxPenStyle, self.ui.comboBoxPenCapStyle, self.ui.comboBoxPenJoinStyle]
        styleMaps = [self.lineStyles, self.capStyles, self.joinStyles]
        for styleMap, cbox in zip(styleMaps, cboxes):
            intData = cbox.itemData(cbox.currentIndex())
            cboxArgs.append(styleMap[intData])
        penWidth = self.ui.doubleSpinBoxPenWidth.value()
        penColor = self.ui.pushButtonPenColor.color()
        self.pen= QtGui.QPen(penColor, penWidth, *cboxArgs)

        displaySize = self.ui.labelPenPreview.size()
        palette = self.ui.labelPenPreview.palette()
        labelBackgroundColor = palette.color(palette.Window)
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
