
# Form implementation generated from reading ui file 'PenSelectorDialogbox.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from ..Qt import QtCore, QtGui
from .ColorButton import ColorButton

    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

__all__ = ['PenSelectorDialogbox']

class PenSelectorDialogbox(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(178, 280)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(178, 280))
        Dialog.setMaximumSize(QtCore.QSize(178, 280))
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)
        self.comboBoxPenJoinStyle = QtGui.QComboBox(Dialog)
        self.comboBoxPenJoinStyle.setObjectName("comboBoxPenJoinStyle")
        self.gridLayout.addWidget(self.comboBoxPenJoinStyle, 5, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)
        self.doubleSpinBoxPenWidth = QtGui.QDoubleSpinBox(Dialog)
        self.doubleSpinBoxPenWidth.setMinimum(0.01)
        self.doubleSpinBoxPenWidth.setValue(1)
        self.doubleSpinBoxPenWidth.setMaximum(9999.99)
        self.doubleSpinBoxPenWidth.setObjectName("doubleSpinBoxPenWidth")
        self.gridLayout.addWidget(self.doubleSpinBoxPenWidth, 0, 1, 1, 1)
        self.comboBoxPenStyle = QtGui.QComboBox(Dialog)
        self.comboBoxPenStyle.setObjectName("comboBoxPenStyle")
        self.gridLayout.addWidget(self.comboBoxPenStyle, 3, 1, 1, 1)
        self.comboBoxPenCapStyle = QtGui.QComboBox(Dialog)
        self.comboBoxPenCapStyle.setObjectName("comboBoxPenCapStyle")
        self.gridLayout.addWidget(self.comboBoxPenCapStyle, 4, 1, 1, 1)
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtGui.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.labelPenPreview = QtGui.QLabel(Dialog)
        self.labelPenPreview.setMinimumSize(QtCore.QSize(160, 64))
        self.labelPenPreview.setMaximumSize(QtCore.QSize(160, 64))
        self.labelPenPreview.setText("")
        self.labelPenPreview.setObjectName("labelPenPreview")
        self.gridLayout.addWidget(self.labelPenPreview, 6, 0, 1, 2)
        self.buttonBoxAcceptCancel = QtGui.QDialogButtonBox(Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBoxAcceptCancel.sizePolicy().hasHeightForWidth())
        self.buttonBoxAcceptCancel.setSizePolicy(sizePolicy)
        self.buttonBoxAcceptCancel.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBoxAcceptCancel.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBoxAcceptCancel.setObjectName("buttonBoxAcceptCancel")
        self.gridLayout.addWidget(self.buttonBoxAcceptCancel, 7, 0, 1, 2)
        self.pushButtonPenColor = ColorButton(Dialog)
        self.pushButtonPenColor.setMinimumSize(QtCore.QSize(0, 24))
        self.pushButtonPenColor.setMaximumSize(QtCore.QSize(16777215, 24))
        self.pushButtonPenColor.setText("")
        self.pushButtonPenColor.setObjectName("pushButtonPenColor")
        self.gridLayout.addWidget(self.pushButtonPenColor, 1, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBoxAcceptCancel, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBoxAcceptCancel, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle("Dialog")
        self.label_3.setText("Pen cap style")
        self.label_2.setText("Pen style")
        self.label_4.setText("Pen join style")
        self.label.setText(_translate("Dialog", "Pen width", None))
        self.label_5.setText(_translate("Dialog", "Pen color", None))

from pyqtgraph.widgets.ColorButton import ColorButton

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
