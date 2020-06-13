# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'examples/designerExample.ui'
#
# Created: Fri Feb 16 20:31:04 2018
#      by: pyside2-uic 2.0.0 running on PySide2 2.0.0~alpha0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.plotBtn = QtWidgets.QPushButton(Form)
        self.plotBtn.setObjectName("plotBtn")
        self.gridLayout.addWidget(self.plotBtn, 0, 0, 1, 1)
        self.plot = PlotWidget(Form)
        self.plot.setObjectName("plot")
        self.gridLayout.addWidget(self.plot, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.plotBtn.setText(QtWidgets.QApplication.translate("Form", "Plot!", None, -1))

from pyqtgraph import PlotWidget
