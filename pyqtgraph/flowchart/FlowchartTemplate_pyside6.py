# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'FlowchartTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ..widgets.DataTreeWidget import DataTreeWidget
from ..flowchart.FlowchartGraphicsView import FlowchartGraphicsView


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(529, 329)
        self.selInfoWidget = QWidget(Form)
        self.selInfoWidget.setObjectName(u"selInfoWidget")
        self.selInfoWidget.setGeometry(QRect(260, 10, 264, 222))
        self.gridLayout = QGridLayout(self.selInfoWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.selDescLabel = QLabel(self.selInfoWidget)
        self.selDescLabel.setObjectName(u"selDescLabel")
        self.selDescLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.selDescLabel.setWordWrap(True)

        self.gridLayout.addWidget(self.selDescLabel, 0, 0, 1, 1)

        self.selNameLabel = QLabel(self.selInfoWidget)
        self.selNameLabel.setObjectName(u"selNameLabel")
        font = QFont()
        font.setBold(True)
        self.selNameLabel.setFont(font)

        self.gridLayout.addWidget(self.selNameLabel, 0, 1, 1, 1)

        self.selectedTree = DataTreeWidget(self.selInfoWidget)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.selectedTree.setHeaderItem(__qtreewidgetitem)
        self.selectedTree.setObjectName(u"selectedTree")

        self.gridLayout.addWidget(self.selectedTree, 1, 0, 1, 2)

        self.hoverText = QTextEdit(Form)
        self.hoverText.setObjectName(u"hoverText")
        self.hoverText.setGeometry(QRect(0, 240, 521, 81))
        self.view = FlowchartGraphicsView(Form)
        self.view.setObjectName(u"view")
        self.view.setGeometry(QRect(0, 0, 256, 192))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.selDescLabel.setText("")
        self.selNameLabel.setText("")
    # retranslateUi

