# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyqtgraph/graphicsItems/lutCtrlTemplate.ui'
#
# Created: Thu Mar  8 09:59:38 2018
#      by: PyQt4 UI code generator 4.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
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

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(200, 62)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.manualRadio = QtGui.QRadioButton(Form)
        self.manualRadio.setObjectName(_fromUtf8("manualRadio"))
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)
        self.logCheck = QtGui.QCheckBox(Form)
        self.logCheck.setObjectName(_fromUtf8("logCheck"))
        self.gridLayout.addWidget(self.logCheck, 3, 0, 1, 4)
        self.autoRadio = QtGui.QRadioButton(Form)
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName(_fromUtf8("autoRadio"))
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)
        self.minText = QtGui.QLineEdit(Form)
        self.minText.setObjectName(_fromUtf8("minText"))
        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)
        self.maxText = QtGui.QLineEdit(Form)
        self.maxText.setObjectName(_fromUtf8("maxText"))
        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.manualRadio.setToolTip(_translate("Form", "<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None))
        self.manualRadio.setText(_translate("Form", "Manual", None))
        self.logCheck.setToolTip(_translate("Form", "<html><head/><body><p>Use log scale</p></body></html>", None))
        self.logCheck.setText(_translate("Form", "Log scale", None))
        self.autoRadio.setToolTip(_translate("Form", "<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None))
        self.autoRadio.setText(_translate("Form", "Auto", None))
        self.minText.setToolTip(_translate("Form", "<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None))
        self.minText.setText(_translate("Form", "0", None))
        self.maxText.setToolTip(_translate("Form", "<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None))
        self.maxText.setText(_translate("Form", "0", None))

