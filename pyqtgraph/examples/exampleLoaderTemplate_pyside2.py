
################################################################################
## Form generated from reading UI file 'exampleLoaderTemplate.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


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
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.gridLayout = QGridLayout(self.layoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.qtLibCombo = QComboBox(self.layoutWidget)
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.setObjectName(u"qtLibCombo")

        self.gridLayout.addWidget(self.qtLibCombo, 4, 1, 1, 1)

        self.loadBtn = QPushButton(self.layoutWidget)
        self.loadBtn.setObjectName(u"loadBtn")

        self.gridLayout.addWidget(self.loadBtn, 6, 1, 1, 1)

        self.exampleTree = QTreeWidget(self.layoutWidget)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.exampleTree.setHeaderItem(__qtreewidgetitem)
        self.exampleTree.setObjectName(u"exampleTree")
        self.exampleTree.header().setVisible(False)

        self.gridLayout.addWidget(self.exampleTree, 3, 0, 1, 2)

        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)

        self.exampleFilter = QLineEdit(self.layoutWidget)
        self.exampleFilter.setObjectName(u"exampleFilter")

        self.gridLayout.addWidget(self.exampleFilter, 0, 0, 1, 2)

        self.searchFiles = QComboBox(self.layoutWidget)
        self.searchFiles.addItem("")
        self.searchFiles.addItem("")
        self.searchFiles.setObjectName(u"searchFiles")

        self.gridLayout.addWidget(self.searchFiles, 1, 0, 1, 2)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.loadedFileLabel = QLabel(self.layoutWidget1)
        self.loadedFileLabel.setObjectName(u"loadedFileLabel")
        font = QFont()
        font.setBold(True)
        self.loadedFileLabel.setFont(font)
        self.loadedFileLabel.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.loadedFileLabel)

        self.codeView = QPlainTextEdit(self.layoutWidget1)
        self.codeView.setObjectName(u"codeView")
        font1 = QFont()
        font1.setFamily(u"Courier New")
        self.codeView.setFont(font1)

        self.verticalLayout.addWidget(self.codeView)

        self.splitter.addWidget(self.layoutWidget1)

        self.gridLayout_2.addWidget(self.splitter, 1, 0, 1, 1)


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

        self.loadBtn.setText(QCoreApplication.translate("Form", u"Run Example", None))
        self.label.setText(QCoreApplication.translate("Form", u"Qt Library:", None))
        self.exampleFilter.setPlaceholderText(QCoreApplication.translate("Form", u"Type to filter...", None))
        self.searchFiles.setItemText(0, QCoreApplication.translate("Form", u"Title Search", None))
        self.searchFiles.setItemText(1, QCoreApplication.translate("Form", u"Content Search", None))

        self.loadedFileLabel.setText("")
    # retranslateUi
