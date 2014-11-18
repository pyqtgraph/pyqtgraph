# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './examples/ScatterPlotSpeedTestTemplate.ui'
#
# Created: Tue Nov 18 09:45:21 2014
#      by: PyQt5 UI code generator 5.1.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

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
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pixelModeCheck.setText(_translate("Form", "pixel mode"))
        self.label.setText(_translate("Form", "Size"))
        self.randCheck.setText(_translate("Form", "Randomize"))

from pyqtgraph import PlotWidget
