# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/flowchart/FlowchartTemplate.ui',
# licensing of 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/flowchart/FlowchartTemplate.ui' applies.
#
# Created: Wed Jan 30 12:16:53 2019
#      by: pyside2-uic  running on PySide2 5.12.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(529, 329)
        self.selInfoWidget = QtWidgets.QWidget(Form)
        self.selInfoWidget.setGeometry(QtCore.QRect(260, 10, 264, 222))
        self.selInfoWidget.setObjectName("selInfoWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.selInfoWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.selDescLabel = QtWidgets.QLabel(self.selInfoWidget)
        self.selDescLabel.setText("")
        self.selDescLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.selDescLabel.setWordWrap(True)
        self.selDescLabel.setObjectName("selDescLabel")
        self.gridLayout.addWidget(self.selDescLabel, 0, 0, 1, 1)
        self.selNameLabel = QtWidgets.QLabel(self.selInfoWidget)
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.selNameLabel.setFont(font)
        self.selNameLabel.setText("")
        self.selNameLabel.setObjectName("selNameLabel")
        self.gridLayout.addWidget(self.selNameLabel, 0, 1, 1, 1)
        self.selectedTree = DataTreeWidget(self.selInfoWidget)
        self.selectedTree.setObjectName("selectedTree")
        self.selectedTree.headerItem().setText(0, "1")
        self.gridLayout.addWidget(self.selectedTree, 1, 0, 1, 2)
        self.hoverText = QtWidgets.QTextEdit(Form)
        self.hoverText.setGeometry(QtCore.QRect(0, 240, 521, 81))
        self.hoverText.setObjectName("hoverText")
        self.view = FlowchartGraphicsView(Form)
        self.view.setGeometry(QtCore.QRect(0, 0, 256, 192))
        self.view.setObjectName("view")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))

from ..widgets.DataTreeWidget import DataTreeWidget
from ..flowchart.FlowchartGraphicsView import FlowchartGraphicsView
