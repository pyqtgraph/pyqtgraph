# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyqtgraph/graphicsItems/lutCtrlTemplate.ui'
#
# Created: Thu Mar  8 09:59:39 2018
#      by: pyside2-uic  running on PySide2 2.0.0~alpha0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(200, 62)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.manualRadio = QtWidgets.QRadioButton(Form)
        self.manualRadio.setObjectName("manualRadio")
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)
        self.logCheck = QtWidgets.QCheckBox(Form)
        self.logCheck.setObjectName("logCheck")
        self.gridLayout.addWidget(self.logCheck, 3, 0, 1, 4)
        self.autoRadio = QtWidgets.QRadioButton(Form)
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName("autoRadio")
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)
        self.minText = QtWidgets.QLineEdit(Form)
        self.minText.setObjectName("minText")
        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)
        self.maxText = QtWidgets.QLineEdit(Form)
        self.maxText.setObjectName("maxText")
        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.manualRadio.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None, -1))
        self.manualRadio.setText(QtWidgets.QApplication.translate("Form", "Manual", None, -1))
        self.logCheck.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Use log scale</p></body></html>", None, -1))
        self.logCheck.setText(QtWidgets.QApplication.translate("Form", "Log scale", None, -1))
        self.autoRadio.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None, -1))
        self.autoRadio.setText(QtWidgets.QApplication.translate("Form", "Auto", None, -1))
        self.minText.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None, -1))
        self.minText.setText(QtWidgets.QApplication.translate("Form", "0", None, -1))
        self.maxText.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None, -1))
        self.maxText.setText(QtWidgets.QApplication.translate("Form", "0", None, -1))

