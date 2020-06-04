# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CanvasTemplate.ui'
#
# Created: Sun Sep 18 19:18:22 2016
#      by: pyside2-uic  running on PySide2 2.0.0~alpha0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(490, 414)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.view = GraphicsView(self.splitter)
        self.view.setObjectName("view")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.autoRangeBtn = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.autoRangeBtn.sizePolicy().hasHeightForWidth())
        self.autoRangeBtn.setSizePolicy(sizePolicy)
        self.autoRangeBtn.setObjectName("autoRangeBtn")
        self.gridLayout_2.addWidget(self.autoRangeBtn, 2, 0, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.redirectCheck = QtWidgets.QCheckBox(self.layoutWidget)
        self.redirectCheck.setObjectName("redirectCheck")
        self.horizontalLayout.addWidget(self.redirectCheck)
        self.redirectCombo = CanvasCombo(self.layoutWidget)
        self.redirectCombo.setObjectName("redirectCombo")
        self.horizontalLayout.addWidget(self.redirectCombo)
        self.gridLayout_2.addLayout(self.horizontalLayout, 5, 0, 1, 2)
        self.itemList = TreeWidget(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.itemList.sizePolicy().hasHeightForWidth())
        self.itemList.setSizePolicy(sizePolicy)
        self.itemList.setHeaderHidden(True)
        self.itemList.setObjectName("itemList")
        self.itemList.headerItem().setText(0, "1")
        self.gridLayout_2.addWidget(self.itemList, 6, 0, 1, 2)
        self.ctrlLayout = QtWidgets.QGridLayout()
        self.ctrlLayout.setSpacing(0)
        self.ctrlLayout.setObjectName("ctrlLayout")
        self.gridLayout_2.addLayout(self.ctrlLayout, 10, 0, 1, 2)
        self.resetTransformsBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.resetTransformsBtn.setObjectName("resetTransformsBtn")
        self.gridLayout_2.addWidget(self.resetTransformsBtn, 7, 0, 1, 1)
        self.mirrorSelectionBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.mirrorSelectionBtn.setObjectName("mirrorSelectionBtn")
        self.gridLayout_2.addWidget(self.mirrorSelectionBtn, 3, 0, 1, 1)
        self.reflectSelectionBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.reflectSelectionBtn.setObjectName("reflectSelectionBtn")
        self.gridLayout_2.addWidget(self.reflectSelectionBtn, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.autoRangeBtn.setText(QtWidgets.QApplication.translate("Form", "Auto Range", None, -1))
        self.redirectCheck.setToolTip(QtWidgets.QApplication.translate("Form", "Check to display all local items in a remote canvas.", None, -1))
        self.redirectCheck.setText(QtWidgets.QApplication.translate("Form", "Redirect", None, -1))
        self.resetTransformsBtn.setText(QtWidgets.QApplication.translate("Form", "Reset Transforms", None, -1))
        self.mirrorSelectionBtn.setText(QtWidgets.QApplication.translate("Form", "Mirror Selection", None, -1))
        self.reflectSelectionBtn.setText(QtWidgets.QApplication.translate("Form", "MirrorXY", None, -1))

from ..widgets.TreeWidget import TreeWidget
from CanvasManager import CanvasCombo
from ..widgets.GraphicsView import GraphicsView
