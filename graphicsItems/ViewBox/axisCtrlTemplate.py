# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'axisCtrlTemplate.ui'
#
# Created: Thu Mar 22 13:13:14 2012
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(182, 120)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.mouseCheck = QtGui.QCheckBox(Form)
        self.mouseCheck.setText(QtGui.QApplication.translate("Form", "Mouse Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.mouseCheck.setChecked(True)
        self.mouseCheck.setObjectName(_fromUtf8("mouseCheck"))
        self.gridLayout.addWidget(self.mouseCheck, 0, 1, 1, 2)
        self.manualRadio = QtGui.QRadioButton(Form)
        self.manualRadio.setText(QtGui.QApplication.translate("Form", "Manual", None, QtGui.QApplication.UnicodeUTF8))
        self.manualRadio.setObjectName(_fromUtf8("manualRadio"))
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 1)
        self.minText = QtGui.QLineEdit(Form)
        self.minText.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.minText.setObjectName(_fromUtf8("minText"))
        self.gridLayout.addWidget(self.minText, 1, 1, 1, 1)
        self.maxText = QtGui.QLineEdit(Form)
        self.maxText.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.maxText.setObjectName(_fromUtf8("maxText"))
        self.gridLayout.addWidget(self.maxText, 1, 2, 1, 1)
        self.autoRadio = QtGui.QRadioButton(Form)
        self.autoRadio.setText(QtGui.QApplication.translate("Form", "Auto", None, QtGui.QApplication.UnicodeUTF8))
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName(_fromUtf8("autoRadio"))
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 1)
        self.autoPercentSpin = QtGui.QSpinBox(Form)
        self.autoPercentSpin.setEnabled(True)
        self.autoPercentSpin.setSuffix(QtGui.QApplication.translate("Form", "%", None, QtGui.QApplication.UnicodeUTF8))
        self.autoPercentSpin.setMinimum(1)
        self.autoPercentSpin.setMaximum(100)
        self.autoPercentSpin.setSingleStep(1)
        self.autoPercentSpin.setProperty("value", 100)
        self.autoPercentSpin.setObjectName(_fromUtf8("autoPercentSpin"))
        self.gridLayout.addWidget(self.autoPercentSpin, 2, 1, 1, 2)
        self.autoPanCheck = QtGui.QCheckBox(Form)
        self.autoPanCheck.setText(QtGui.QApplication.translate("Form", "Auto Pan Only", None, QtGui.QApplication.UnicodeUTF8))
        self.autoPanCheck.setObjectName(_fromUtf8("autoPanCheck"))
        self.gridLayout.addWidget(self.autoPanCheck, 3, 1, 1, 2)
        self.linkCombo = QtGui.QComboBox(Form)
        self.linkCombo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)
        self.linkCombo.setObjectName(_fromUtf8("linkCombo"))
        self.gridLayout.addWidget(self.linkCombo, 4, 1, 1, 2)
        self.label = QtGui.QLabel(Form)
        self.label.setText(QtGui.QApplication.translate("Form", "Link Axis:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        pass

