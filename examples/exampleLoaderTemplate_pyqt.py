# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'examples/exampleLoaderTemplate.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(846, 552)
        self.gridLayout_2 = QtGui.QGridLayout(Form)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.widget = QtGui.QWidget(self.splitter)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout = QtGui.QGridLayout(self.widget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.exampleTree = QtGui.QTreeWidget(self.widget)
        self.exampleTree.setObjectName(_fromUtf8("exampleTree"))
        self.exampleTree.headerItem().setText(0, _fromUtf8("1"))
        self.exampleTree.header().setVisible(False)
        self.gridLayout.addWidget(self.exampleTree, 0, 0, 1, 2)
        self.graphicsSystemCombo = QtGui.QComboBox(self.widget)
        self.graphicsSystemCombo.setObjectName(_fromUtf8("graphicsSystemCombo"))
        self.graphicsSystemCombo.addItem(_fromUtf8(""))
        self.graphicsSystemCombo.addItem(_fromUtf8(""))
        self.graphicsSystemCombo.addItem(_fromUtf8(""))
        self.graphicsSystemCombo.addItem(_fromUtf8(""))
        self.gridLayout.addWidget(self.graphicsSystemCombo, 2, 1, 1, 1)
        self.qtLibCombo = QtGui.QComboBox(self.widget)
        self.qtLibCombo.setObjectName(_fromUtf8("qtLibCombo"))
        self.qtLibCombo.addItem(_fromUtf8(""))
        self.qtLibCombo.addItem(_fromUtf8(""))
        self.qtLibCombo.addItem(_fromUtf8(""))
        self.qtLibCombo.addItem(_fromUtf8(""))
        self.qtLibCombo.addItem(_fromUtf8(""))
        self.gridLayout.addWidget(self.qtLibCombo, 1, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.widget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label = QtGui.QLabel(self.widget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.loadBtn = QtGui.QPushButton(self.widget)
        self.loadBtn.setObjectName(_fromUtf8("loadBtn"))
        self.gridLayout.addWidget(self.loadBtn, 3, 1, 1, 1)
        self.widget1 = QtGui.QWidget(self.splitter)
        self.widget1.setObjectName(_fromUtf8("widget1"))
        self.verticalLayout = QtGui.QVBoxLayout(self.widget1)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.loadedFileLabel = QtGui.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.loadedFileLabel.setFont(font)
        self.loadedFileLabel.setText(_fromUtf8(""))
        self.loadedFileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.loadedFileLabel.setObjectName(_fromUtf8("loadedFileLabel"))
        self.verticalLayout.addWidget(self.loadedFileLabel)
        self.codeView = QtGui.QPlainTextEdit(self.widget1)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("FreeMono"))
        self.codeView.setFont(font)
        self.codeView.setObjectName(_fromUtf8("codeView"))
        self.verticalLayout.addWidget(self.codeView)
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.graphicsSystemCombo.setItemText(0, _translate("Form", "default", None))
        self.graphicsSystemCombo.setItemText(1, _translate("Form", "native", None))
        self.graphicsSystemCombo.setItemText(2, _translate("Form", "raster", None))
        self.graphicsSystemCombo.setItemText(3, _translate("Form", "opengl", None))
        self.qtLibCombo.setItemText(0, _translate("Form", "default", None))
        self.qtLibCombo.setItemText(1, _translate("Form", "PyQt4", None))
        self.qtLibCombo.setItemText(2, _translate("Form", "PySide", None))
        self.qtLibCombo.setItemText(3, _translate("Form", "PyQt5", None))
        self.qtLibCombo.setItemText(4, _translate("Form", "PySide2", None))
        self.label_2.setText(_translate("Form", "Graphics System:", None))
        self.label.setText(_translate("Form", "Qt Library:", None))
        self.loadBtn.setText(_translate("Form", "Run Example", None))

