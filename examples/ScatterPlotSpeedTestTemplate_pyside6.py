# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ScatterPlotSpeedTestTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.0.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from pyqtgraph import PlotWidget


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(400, 300)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.sizeSpin = QSpinBox(Form)
        self.sizeSpin.setObjectName(u"sizeSpin")
        self.sizeSpin.setValue(10)

        self.gridLayout.addWidget(self.sizeSpin, 1, 1, 1, 1)

        self.pixelModeCheck = QCheckBox(Form)
        self.pixelModeCheck.setObjectName(u"pixelModeCheck")

        self.gridLayout.addWidget(self.pixelModeCheck, 1, 3, 1, 1)

        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.plot = PlotWidget(Form)
        self.plot.setObjectName(u"plot")

        self.gridLayout.addWidget(self.plot, 0, 0, 1, 4)

        self.randCheck = QCheckBox(Form)
        self.randCheck.setObjectName(u"randCheck")

        self.gridLayout.addWidget(self.randCheck, 1, 2, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.pixelModeCheck.setText(QCoreApplication.translate("Form", u"pixel mode", None))
        self.label.setText(QCoreApplication.translate("Form", u"Size", None))
        self.randCheck.setText(QCoreApplication.translate("Form", u"Randomize", None))
    # retranslateUi

