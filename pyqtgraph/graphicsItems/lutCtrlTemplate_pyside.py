# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyqtgraph/graphicsItems/lutCtrlTemplate.ui'
#
# Created: Thu Mar  8 09:59:38 2018
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(200, 62)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.manualRadio = QtGui.QRadioButton(Form)
        self.manualRadio.setObjectName("manualRadio")
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)
        self.logCheck = QtGui.QCheckBox(Form)
        self.logCheck.setObjectName("logCheck")
        self.gridLayout.addWidget(self.logCheck, 3, 0, 1, 4)
        self.autoRadio = QtGui.QRadioButton(Form)
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName("autoRadio")
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)
        self.minText = QtGui.QLineEdit(Form)
        self.minText.setObjectName("minText")
        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)
        self.maxText = QtGui.QLineEdit(Form)
        self.maxText.setObjectName("maxText")
        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.manualRadio.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.manualRadio.setText(QtGui.QApplication.translate("Form", "Manual", None, QtGui.QApplication.UnicodeUTF8))
        self.logCheck.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Use log scale</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.logCheck.setText(QtGui.QApplication.translate("Form", "Log scale", None, QtGui.QApplication.UnicodeUTF8))
        self.autoRadio.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.autoRadio.setText(QtGui.QApplication.translate("Form", "Auto", None, QtGui.QApplication.UnicodeUTF8))
        self.minText.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.minText.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.maxText.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.maxText.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))

