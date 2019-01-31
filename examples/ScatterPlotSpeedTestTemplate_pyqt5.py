# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './examples/ScatterPlotSpeedTestTemplate.ui'
#
# Created: Fri Sep 21 15:39:09 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.sizeSpin = QtWidgets.QSpinBox(Form)
        self.sizeSpin.setProperty("value", 10)
        self.sizeSpin.setObjectName(_fromUtf8("sizeSpin"))
        self.gridLayout.addWidget(self.sizeSpin, 1, 1, 1, 1)
        self.pixelModeCheck = QtWidgets.QCheckBox(Form)
        self.pixelModeCheck.setObjectName(_fromUtf8("pixelModeCheck"))
        self.gridLayout.addWidget(self.pixelModeCheck, 1, 3, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.plot = PlotWidget(Form)
        self.plot.setObjectName(_fromUtf8("plot"))
        self.gridLayout.addWidget(self.plot, 0, 0, 1, 4)
        self.randCheck = QtWidgets.QCheckBox(Form)
        self.randCheck.setObjectName(_fromUtf8("randCheck"))
        self.gridLayout.addWidget(self.randCheck, 1, 2, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, QtWidgets.QApplication.UnicodeUTF8))
        self.pixelModeCheck.setText(QtWidgets.QApplication.translate("Form", "pixel mode", None, QtWidgets.QApplication.UnicodeUTF8))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Size", None, QtWidgets.QApplication.UnicodeUTF8))
        self.randCheck.setText(QtWidgets.QApplication.translate("Form", "Randomize", None, QtWidgets.QApplication.UnicodeUTF8))

from pyqtgraph import PlotWidget
