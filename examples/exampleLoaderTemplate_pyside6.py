# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'exampleLoaderTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(846, 552)
        self.gridLayout_2 = QGridLayout(Form)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.exampleTree = QTreeWidget(self.widget)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.exampleTree.setHeaderItem(__qtreewidgetitem)
        self.exampleTree.setObjectName(u"exampleTree")
        self.exampleTree.header().setVisible(False)

        self.gridLayout.addWidget(self.exampleTree, 0, 0, 1, 2)

        self.qtLibCombo = QComboBox(self.widget)
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.setObjectName(u"qtLibCombo")

        self.gridLayout.addWidget(self.qtLibCombo, 1, 1, 1, 1)

        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.loadBtn = QPushButton(self.widget)
        self.loadBtn.setObjectName(u"loadBtn")

        self.gridLayout.addWidget(self.loadBtn, 3, 1, 1, 1)

        self.splitter.addWidget(self.widget)
        self.widget1 = QWidget(self.splitter)
        self.widget1.setObjectName(u"widget1")
        self.verticalLayout = QVBoxLayout(self.widget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.loadedFileLabel = QLabel(self.widget1)
        self.loadedFileLabel.setObjectName(u"loadedFileLabel")
        font = QFont()
        font.setBold(True)
        self.loadedFileLabel.setFont(font)
        self.loadedFileLabel.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.loadedFileLabel)

        self.codeView = QPlainTextEdit(self.widget1)
        self.codeView.setObjectName(u"codeView")
        font1 = QFont()
        font1.setFamilies([u"Courier New"])
        self.codeView.setFont(font1)

        self.verticalLayout.addWidget(self.codeView)

        self.splitter.addWidget(self.widget1)

        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.qtLibCombo.setItemText(0, QCoreApplication.translate("Form", u"default", None))
        self.qtLibCombo.setItemText(1, QCoreApplication.translate("Form", u"PyQt5", None))
        self.qtLibCombo.setItemText(2, QCoreApplication.translate("Form", u"PySide2", None))
        self.qtLibCombo.setItemText(3, QCoreApplication.translate("Form", u"PySide6", None))
        self.qtLibCombo.setItemText(4, QCoreApplication.translate("Form", u"PyQt6", None))

        self.label.setText(QCoreApplication.translate("Form", u"Qt Library:", None))
        self.loadBtn.setText(QCoreApplication.translate("Form", u"Run Example", None))
        self.loadedFileLabel.setText("")
    # retranslateUi

