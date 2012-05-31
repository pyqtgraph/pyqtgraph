# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ScatterPlotSpeedTestTemplate.ui'
#
# Created: Tue May  8 23:09:16 2012
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
        Form.resize(400, 300)
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.plot = PlotWidget(Form)
        self.plot.setObjectName(_fromUtf8("plot"))
        self.gridLayout.addWidget(self.plot, 0, 0, 1, 2)
        self.identicalCheck = QtGui.QCheckBox(Form)
        self.identicalCheck.setText(QtGui.QApplication.translate("Form", "Identical", None, QtGui.QApplication.UnicodeUTF8))
        self.identicalCheck.setObjectName(_fromUtf8("identicalCheck"))
        self.gridLayout.addWidget(self.identicalCheck, 1, 0, 1, 1)
        self.pixelModeCheck = QtGui.QCheckBox(Form)
        self.pixelModeCheck.setText(QtGui.QApplication.translate("Form", "pixel mode", None, QtGui.QApplication.UnicodeUTF8))
        self.pixelModeCheck.setObjectName(_fromUtf8("pixelModeCheck"))
        self.gridLayout.addWidget(self.pixelModeCheck, 1, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        pass

from pyqtgraph import PlotWidget
