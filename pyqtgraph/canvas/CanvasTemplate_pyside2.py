# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/canvas/CanvasTemplate.ui',
# licensing of 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/canvas/CanvasTemplate.ui' applies.
#
# Created: Wed Jan 30 12:16:53 2019
#      by: pyside2-uic  running on PySide2 5.12.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(821, 578)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.view = GraphicsView(self.splitter)
        self.view.setObjectName("view")
        self.vsplitter = QtWidgets.QSplitter(self.splitter)
        self.vsplitter.setOrientation(QtCore.Qt.Vertical)
        self.vsplitter.setObjectName("vsplitter")
        self.canvasCtrlWidget = QtWidgets.QWidget(self.vsplitter)
        self.canvasCtrlWidget.setObjectName("canvasCtrlWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.canvasCtrlWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.autoRangeBtn = QtWidgets.QPushButton(self.canvasCtrlWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.autoRangeBtn.sizePolicy().hasHeightForWidth())
        self.autoRangeBtn.setSizePolicy(sizePolicy)
        self.autoRangeBtn.setObjectName("autoRangeBtn")
        self.gridLayout.addWidget(self.autoRangeBtn, 0, 0, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.redirectCheck = QtWidgets.QCheckBox(self.canvasCtrlWidget)
        self.redirectCheck.setObjectName("redirectCheck")
        self.horizontalLayout.addWidget(self.redirectCheck)
        self.redirectCombo = CanvasCombo(self.canvasCtrlWidget)
        self.redirectCombo.setObjectName("redirectCombo")
        self.horizontalLayout.addWidget(self.redirectCombo)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        self.itemList = TreeWidget(self.canvasCtrlWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.itemList.sizePolicy().hasHeightForWidth())
        self.itemList.setSizePolicy(sizePolicy)
        self.itemList.setHeaderHidden(True)
        self.itemList.setObjectName("itemList")
        self.itemList.headerItem().setText(0, "1")
        self.gridLayout.addWidget(self.itemList, 2, 0, 1, 2)
        self.resetTransformsBtn = QtWidgets.QPushButton(self.canvasCtrlWidget)
        self.resetTransformsBtn.setObjectName("resetTransformsBtn")
        self.gridLayout.addWidget(self.resetTransformsBtn, 3, 0, 1, 2)
        self.mirrorSelectionBtn = QtWidgets.QPushButton(self.canvasCtrlWidget)
        self.mirrorSelectionBtn.setObjectName("mirrorSelectionBtn")
        self.gridLayout.addWidget(self.mirrorSelectionBtn, 4, 0, 1, 1)
        self.reflectSelectionBtn = QtWidgets.QPushButton(self.canvasCtrlWidget)
        self.reflectSelectionBtn.setObjectName("reflectSelectionBtn")
        self.gridLayout.addWidget(self.reflectSelectionBtn, 4, 1, 1, 1)
        self.canvasItemCtrl = QtWidgets.QWidget(self.vsplitter)
        self.canvasItemCtrl.setObjectName("canvasItemCtrl")
        self.ctrlLayout = QtWidgets.QGridLayout(self.canvasItemCtrl)
        self.ctrlLayout.setContentsMargins(0, 0, 0, 0)
        self.ctrlLayout.setSpacing(0)
        self.ctrlLayout.setObjectName("ctrlLayout")
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)

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

from ..widgets.GraphicsView import GraphicsView
from .CanvasManager import CanvasCombo
from ..widgets.TreeWidget import TreeWidget
