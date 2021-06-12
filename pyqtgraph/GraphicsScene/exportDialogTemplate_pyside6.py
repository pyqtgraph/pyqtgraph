# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'exportDialogTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ..parametertree import ParameterTree


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(241, 367)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)

        self.itemTree = QTreeWidget(Form)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.itemTree.setHeaderItem(__qtreewidgetitem)
        self.itemTree.setObjectName(u"itemTree")
        self.itemTree.header().setVisible(False)

        self.gridLayout.addWidget(self.itemTree, 1, 0, 1, 3)

        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 3)

        self.formatList = QListWidget(Form)
        self.formatList.setObjectName(u"formatList")

        self.gridLayout.addWidget(self.formatList, 3, 0, 1, 3)

        self.exportBtn = QPushButton(Form)
        self.exportBtn.setObjectName(u"exportBtn")

        self.gridLayout.addWidget(self.exportBtn, 6, 1, 1, 1)

        self.closeBtn = QPushButton(Form)
        self.closeBtn.setObjectName(u"closeBtn")

        self.gridLayout.addWidget(self.closeBtn, 6, 2, 1, 1)

        self.paramTree = ParameterTree(Form)
        __qtreewidgetitem1 = QTreeWidgetItem()
        __qtreewidgetitem1.setText(0, u"1");
        self.paramTree.setHeaderItem(__qtreewidgetitem1)
        self.paramTree.setObjectName(u"paramTree")
        self.paramTree.setColumnCount(2)
        self.paramTree.header().setVisible(False)

        self.gridLayout.addWidget(self.paramTree, 5, 0, 1, 3)

        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 3)

        self.copyBtn = QPushButton(Form)
        self.copyBtn.setObjectName(u"copyBtn")

        self.gridLayout.addWidget(self.copyBtn, 6, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Export", None))
        self.label.setText(QCoreApplication.translate("Form", u"Item to export:", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Export format", None))
        self.exportBtn.setText(QCoreApplication.translate("Form", u"Export", None))
        self.closeBtn.setText(QCoreApplication.translate("Form", u"Close", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Export options", None))
        self.copyBtn.setText(QCoreApplication.translate("Form", u"Copy", None))
    # retranslateUi

