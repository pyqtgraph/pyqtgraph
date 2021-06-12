# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exampleLoaderTemplate.ui',
# licensing of 'exampleLoaderTemplate.ui' applies.
#
# Created: Mon Feb 22 18:33:36 2021
#      by: pyside2-uic  running on PySide2 5.12.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(846, 552)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.exampleTree = QtWidgets.QTreeWidget(self.widget)
        self.exampleTree.setObjectName("exampleTree")
        self.exampleTree.headerItem().setText(0, "1")
        self.exampleTree.header().setVisible(False)
        self.gridLayout.addWidget(self.exampleTree, 0, 0, 1, 2)
        self.qtLibCombo = QtWidgets.QComboBox(self.widget)
        self.qtLibCombo.setObjectName("qtLibCombo")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.qtLibCombo.addItem("")
        self.gridLayout.addWidget(self.qtLibCombo, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.loadBtn = QtWidgets.QPushButton(self.widget)
        self.loadBtn.setObjectName("loadBtn")
        self.gridLayout.addWidget(self.loadBtn, 3, 1, 1, 1)
        self.widget1 = QtWidgets.QWidget(self.splitter)
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.loadedFileLabel = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setBold(True)
        self.loadedFileLabel.setFont(font)
        self.loadedFileLabel.setText("")
        self.loadedFileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.loadedFileLabel.setObjectName("loadedFileLabel")
        self.verticalLayout.addWidget(self.loadedFileLabel)
        self.codeView = QtWidgets.QPlainTextEdit(self.widget1)
        font = QtGui.QFont()
        font.setFamily("Courier New")
        self.codeView.setFont(font)
        self.codeView.setObjectName("codeView")
        self.verticalLayout.addWidget(self.codeView)
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "PyQtGraph", None, -1))
        self.qtLibCombo.setItemText(0, QtWidgets.QApplication.translate("Form", "default", None, -1))
        self.qtLibCombo.setItemText(1, QtWidgets.QApplication.translate("Form", "PyQt5", None, -1))
        self.qtLibCombo.setItemText(2, QtWidgets.QApplication.translate("Form", "PySide2", None, -1))
        self.qtLibCombo.setItemText(3, QtWidgets.QApplication.translate("Form", "PySide6", None, -1))
        self.qtLibCombo.setItemText(4, QtWidgets.QApplication.translate("Form", "PyQt6", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Qt Library:", None, -1))
        self.loadBtn.setText(QtWidgets.QApplication.translate("Form", "Run Example", None, -1))

