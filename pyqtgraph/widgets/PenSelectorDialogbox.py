
# Form implementation generated from reading ui file 'PenSelectorDialogbox.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from ..Qt import QtCore, QtWidgets
from .ColorButton import ColorButton

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

__all__ = ['PenSelectorDialogbox']

class PenSelectorDialogbox:
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(178, 280)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(178, 280))
        Dialog.setMaximumSize(QtCore.QSize(178, 280))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)
        self.comboBoxPenJoinStyle = QtWidgets.QComboBox(Dialog)
        self.comboBoxPenJoinStyle.setObjectName("comboBoxPenJoinStyle")
        self.gridLayout.addWidget(self.comboBoxPenJoinStyle, 5, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)
        self.doubleSpinBoxPenWidth = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBoxPenWidth.setMinimum(0.01)
        self.doubleSpinBoxPenWidth.setValue(1)
        self.doubleSpinBoxPenWidth.setMaximum(9999.99)
        self.doubleSpinBoxPenWidth.setObjectName("doubleSpinBoxPenWidth")
        self.gridLayout.addWidget(self.doubleSpinBoxPenWidth, 0, 1, 1, 1)
        self.comboBoxPenStyle = QtWidgets.QComboBox(Dialog)
        self.comboBoxPenStyle.setObjectName("comboBoxPenStyle")
        self.gridLayout.addWidget(self.comboBoxPenStyle, 3, 1, 1, 1)
        self.comboBoxPenCapStyle = QtWidgets.QComboBox(Dialog)
        self.comboBoxPenCapStyle.setObjectName("comboBoxPenCapStyle")
        self.gridLayout.addWidget(self.comboBoxPenCapStyle, 4, 1, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.labelPenPreview = QtWidgets.QLabel(Dialog)
        self.labelPenPreview.setMinimumSize(QtCore.QSize(160, 64))
        self.labelPenPreview.setMaximumSize(QtCore.QSize(160, 64))
        self.labelPenPreview.setText("")
        self.labelPenPreview.setObjectName("labelPenPreview")
        self.gridLayout.addWidget(self.labelPenPreview, 6, 0, 1, 2)
        self.buttonBoxAcceptCancel = QtWidgets.QDialogButtonBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBoxAcceptCancel.sizePolicy().hasHeightForWidth())
        self.buttonBoxAcceptCancel.setSizePolicy(sizePolicy)
        self.buttonBoxAcceptCancel.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBoxAcceptCancel.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBoxAcceptCancel.setObjectName("buttonBoxAcceptCancel")
        self.gridLayout.addWidget(self.buttonBoxAcceptCancel, 7, 0, 1, 2)
        self.pushButtonPenColor = ColorButton(Dialog)
        self.pushButtonPenColor.setMinimumSize(QtCore.QSize(0, 24))
        self.pushButtonPenColor.setMaximumSize(QtCore.QSize(16777215, 24))
        self.pushButtonPenColor.setText("")
        self.pushButtonPenColor.setObjectName("pushButtonPenColor")
        self.gridLayout.addWidget(self.pushButtonPenColor, 1, 1, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBoxAcceptCancel.accepted.connect(Dialog.accept)
        self.buttonBoxAcceptCancel.rejected.connect(Dialog.reject)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle("Dialog")
        self.label_3.setText("Pen cap style")
        self.label_2.setText("Pen style")
        self.label_4.setText("Pen join style")
        self.label.setText(_translate("Dialog", "Pen width", None))
        self.label_5.setText(_translate("Dialog", "Pen color", None))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = PenSelectorDialogbox()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
