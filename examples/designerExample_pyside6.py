# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designerExample.ui'
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
        self.plotBtn = QPushButton(Form)
        self.plotBtn.setObjectName(u"plotBtn")

        self.gridLayout.addWidget(self.plotBtn, 0, 0, 1, 1)

        self.plot = PlotWidget(Form)
        self.plot.setObjectName(u"plot")

        self.gridLayout.addWidget(self.plot, 1, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.plotBtn.setText(QCoreApplication.translate("Form", u"Plot!", None))
    # retranslateUi

