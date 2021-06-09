# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'FlowchartCtrlTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ..widgets.TreeWidget import TreeWidget
from ..widgets.FeedbackButton import FeedbackButton


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(217, 499)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setVerticalSpacing(0)
        self.loadBtn = QPushButton(Form)
        self.loadBtn.setObjectName(u"loadBtn")

        self.gridLayout.addWidget(self.loadBtn, 1, 0, 1, 1)

        self.saveBtn = FeedbackButton(Form)
        self.saveBtn.setObjectName(u"saveBtn")

        self.gridLayout.addWidget(self.saveBtn, 1, 1, 1, 2)

        self.saveAsBtn = FeedbackButton(Form)
        self.saveAsBtn.setObjectName(u"saveAsBtn")

        self.gridLayout.addWidget(self.saveAsBtn, 1, 3, 1, 1)

        self.reloadBtn = FeedbackButton(Form)
        self.reloadBtn.setObjectName(u"reloadBtn")
        self.reloadBtn.setCheckable(False)
        self.reloadBtn.setFlat(False)

        self.gridLayout.addWidget(self.reloadBtn, 4, 0, 1, 2)

        self.showChartBtn = QPushButton(Form)
        self.showChartBtn.setObjectName(u"showChartBtn")
        self.showChartBtn.setCheckable(True)

        self.gridLayout.addWidget(self.showChartBtn, 4, 2, 1, 2)

        self.ctrlList = TreeWidget(Form)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.ctrlList.setHeaderItem(__qtreewidgetitem)
        self.ctrlList.setObjectName(u"ctrlList")
        self.ctrlList.header().setVisible(False)
        self.ctrlList.header().setStretchLastSection(False)

        self.gridLayout.addWidget(self.ctrlList, 3, 0, 1, 4)

        self.fileNameLabel = QLabel(Form)
        self.fileNameLabel.setObjectName(u"fileNameLabel")
        font = QFont()
        font.setBold(True)
        self.fileNameLabel.setFont(font)
        self.fileNameLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.fileNameLabel, 0, 1, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.loadBtn.setText(QCoreApplication.translate("Form", u"Load..", None))
        self.saveBtn.setText(QCoreApplication.translate("Form", u"Save", None))
        self.saveAsBtn.setText(QCoreApplication.translate("Form", u"As..", None))
        self.reloadBtn.setText(QCoreApplication.translate("Form", u"Reload Libs", None))
        self.showChartBtn.setText(QCoreApplication.translate("Form", u"Flowchart", None))
        self.fileNameLabel.setText("")
    # retranslateUi

