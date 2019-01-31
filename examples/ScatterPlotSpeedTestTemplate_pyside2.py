# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './examples/ScatterPlotSpeedTestTemplate.ui'
#
# Created: Fri Sep 21 15:39:09 2012
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.sizeSpin = QtWidgets.QSpinBox(Form)
        self.sizeSpin.setProperty("value", 10)
        self.sizeSpin.setObjectName("sizeSpin")
        self.gridLayout.addWidget(self.sizeSpin, 1, 1, 1, 1)
        self.pixelModeCheck = QtWidgets.QCheckBox(Form)
        self.pixelModeCheck.setObjectName("pixelModeCheck")
        self.gridLayout.addWidget(self.pixelModeCheck, 1, 3, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.plot = PlotWidget(Form)
        self.plot.setObjectName("plot")
        self.gridLayout.addWidget(self.plot, 0, 0, 1, 4)
        self.randCheck = QtWidgets.QCheckBox(Form)
        self.randCheck.setObjectName("randCheck")
        self.gridLayout.addWidget(self.randCheck, 1, 2, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None)) # , QtWidgets.QApplication.UnicodeUTF8))
        self.pixelModeCheck.setText(QtWidgets.QApplication.translate("Form", "pixel mode", None))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Size", None))
        self.randCheck.setText(QtWidgets.QApplication.translate("Form", "Randomize", None))

from pyqtgraph import PlotWidget
