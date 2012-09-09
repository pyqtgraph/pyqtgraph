# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './examples/ScatterPlotSpeedTestTemplate.ui'
#
# Created: Sun Sep  9 14:41:31 2012
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.plot = PlotWidget(Form)
        self.plot.setObjectName("plot")
        self.gridLayout.addWidget(self.plot, 0, 0, 1, 2)
        self.identicalCheck = QtGui.QCheckBox(Form)
        self.identicalCheck.setObjectName("identicalCheck")
        self.gridLayout.addWidget(self.identicalCheck, 1, 0, 1, 1)
        self.pixelModeCheck = QtGui.QCheckBox(Form)
        self.pixelModeCheck.setObjectName("pixelModeCheck")
        self.gridLayout.addWidget(self.pixelModeCheck, 1, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.identicalCheck.setText(QtGui.QApplication.translate("Form", "Identical", None, QtGui.QApplication.UnicodeUTF8))
        self.pixelModeCheck.setText(QtGui.QApplication.translate("Form", "pixel mode", None, QtGui.QApplication.UnicodeUTF8))

from pyqtgraph import PlotWidget
