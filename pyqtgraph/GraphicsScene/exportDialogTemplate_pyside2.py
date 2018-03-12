# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './pyqtgraph/GraphicsScene/exportDialogTemplate.ui'
#
# Created: Thu Jun  1 08:59:31 2017
#      by: pyside2-uic  running on PySide2 2.0.0~alpha0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(241, 367)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)
        self.itemTree = QtWidgets.QTreeWidget(Form)
        self.itemTree.setObjectName("itemTree")
        self.itemTree.headerItem().setText(0, "1")
        self.itemTree.header().setVisible(False)
        self.gridLayout.addWidget(self.itemTree, 1, 0, 1, 3)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 3)
        self.formatList = QtWidgets.QListWidget(Form)
        self.formatList.setObjectName("formatList")
        self.gridLayout.addWidget(self.formatList, 3, 0, 1, 3)
        self.exportBtn = QtWidgets.QPushButton(Form)
        self.exportBtn.setObjectName("exportBtn")
        self.gridLayout.addWidget(self.exportBtn, 6, 1, 1, 1)
        self.closeBtn = QtWidgets.QPushButton(Form)
        self.closeBtn.setObjectName("closeBtn")
        self.gridLayout.addWidget(self.closeBtn, 6, 2, 1, 1)
        self.paramTree = ParameterTree(Form)
        self.paramTree.setObjectName("paramTree")
        self.paramTree.headerItem().setText(0, "1")
        self.paramTree.header().setVisible(False)
        self.gridLayout.addWidget(self.paramTree, 5, 0, 1, 3)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 3)
        self.copyBtn = QtWidgets.QPushButton(Form)
        self.copyBtn.setObjectName("copyBtn")
        self.gridLayout.addWidget(self.copyBtn, 6, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Export", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Item to export:", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("Form", "Export format", None, -1))
        self.exportBtn.setText(QtWidgets.QApplication.translate("Form", "Export", None, -1))
        self.closeBtn.setText(QtWidgets.QApplication.translate("Form", "Close", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("Form", "Export options", None, -1))
        self.copyBtn.setText(QtWidgets.QApplication.translate("Form", "Copy", None, -1))

from ..parametertree import ParameterTree
