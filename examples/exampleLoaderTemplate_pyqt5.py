# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exampleLoaderTemplate.ui'
#
# Created: Sat Feb 28 09:38:17 2015
#      by: PyQt5 UI code generator 5.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(623, 380)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.exampleTree = QtWidgets.QTreeWidget(self.widget)
        self.exampleTree.setObjectName("exampleTree")
        self.exampleTree.headerItem().setText(0, "1")
        self.exampleTree.header().setVisible(False)
        self.verticalLayout.addWidget(self.exampleTree)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pyqtCheck = QtWidgets.QCheckBox(self.widget)
        self.pyqtCheck.setObjectName("pyqtCheck")
        self.horizontalLayout.addWidget(self.pyqtCheck)
        self.pysideCheck = QtWidgets.QCheckBox(self.widget)
        self.pysideCheck.setObjectName("pysideCheck")
        self.horizontalLayout.addWidget(self.pysideCheck)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.forceGraphicsCheck = QtWidgets.QCheckBox(self.widget)
        self.forceGraphicsCheck.setObjectName("forceGraphicsCheck")
        self.horizontalLayout_2.addWidget(self.forceGraphicsCheck)
        self.forceGraphicsCombo = QtWidgets.QComboBox(self.widget)
        self.forceGraphicsCombo.setObjectName("forceGraphicsCombo")
        self.forceGraphicsCombo.addItem("")
        self.forceGraphicsCombo.addItem("")
        self.forceGraphicsCombo.addItem("")
        self.horizontalLayout_2.addWidget(self.forceGraphicsCombo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.loadBtn = QtWidgets.QPushButton(self.widget)
        self.loadBtn.setObjectName("loadBtn")
        self.verticalLayout.addWidget(self.loadBtn)
        self.widget1 = QtWidgets.QWidget(self.splitter)
        self.widget1.setObjectName("widget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.loadedFileLabel = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.loadedFileLabel.setFont(font)
        self.loadedFileLabel.setText("")
        self.loadedFileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.loadedFileLabel.setObjectName("loadedFileLabel")
        self.verticalLayout_2.addWidget(self.loadedFileLabel)
        self.codeView = QtWidgets.QPlainTextEdit(self.widget1)
        font = QtGui.QFont()
        font.setFamily("FreeMono")
        self.codeView.setFont(font)
        self.codeView.setObjectName("codeView")
        self.verticalLayout_2.addWidget(self.codeView)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pyqtCheck.setText(_translate("Form", "Force PyQt"))
        self.pysideCheck.setText(_translate("Form", "Force PySide"))
        self.forceGraphicsCheck.setText(_translate("Form", "Force Graphics System:"))
        self.forceGraphicsCombo.setItemText(0, _translate("Form", "native"))
        self.forceGraphicsCombo.setItemText(1, _translate("Form", "raster"))
        self.forceGraphicsCombo.setItemText(2, _translate("Form", "opengl"))
        self.loadBtn.setText(_translate("Form", "Run Example"))

