# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CanvasTemplate.ui'
#
# Created: Fri Mar 24 16:09:39 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(821, 578)
        self.gridLayout_2 = QtGui.QGridLayout(Form)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.view = GraphicsView(self.splitter)
        self.view.setObjectName("view")
        self.vsplitter = QtGui.QSplitter(self.splitter)
        self.vsplitter.setOrientation(QtCore.Qt.Vertical)
        self.vsplitter.setObjectName("vsplitter")
        self.canvasCtrlWidget = QtGui.QWidget(self.vsplitter)
        self.canvasCtrlWidget.setObjectName("canvasCtrlWidget")
        self.gridLayout = QtGui.QGridLayout(self.canvasCtrlWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.autoRangeBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.autoRangeBtn.sizePolicy().hasHeightForWidth())
        self.autoRangeBtn.setSizePolicy(sizePolicy)
        self.autoRangeBtn.setObjectName("autoRangeBtn")
        self.gridLayout.addWidget(self.autoRangeBtn, 0, 0, 1, 2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.redirectCheck = QtGui.QCheckBox(self.canvasCtrlWidget)
        self.redirectCheck.setObjectName("redirectCheck")
        self.horizontalLayout.addWidget(self.redirectCheck)
        self.redirectCombo = CanvasCombo(self.canvasCtrlWidget)
        self.redirectCombo.setObjectName("redirectCombo")
        self.horizontalLayout.addWidget(self.redirectCombo)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        self.itemList = TreeWidget(self.canvasCtrlWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.itemList.sizePolicy().hasHeightForWidth())
        self.itemList.setSizePolicy(sizePolicy)
        self.itemList.setHeaderHidden(True)
        self.itemList.setObjectName("itemList")
        self.itemList.headerItem().setText(0, "1")
        self.gridLayout.addWidget(self.itemList, 2, 0, 1, 2)
        self.resetTransformsBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        self.resetTransformsBtn.setObjectName("resetTransformsBtn")
        self.gridLayout.addWidget(self.resetTransformsBtn, 3, 0, 1, 2)
        self.mirrorSelectionBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        self.mirrorSelectionBtn.setObjectName("mirrorSelectionBtn")
        self.gridLayout.addWidget(self.mirrorSelectionBtn, 4, 0, 1, 1)
        self.reflectSelectionBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        self.reflectSelectionBtn.setObjectName("reflectSelectionBtn")
        self.gridLayout.addWidget(self.reflectSelectionBtn, 4, 1, 1, 1)
        self.canvasItemCtrl = QtGui.QWidget(self.vsplitter)
        self.canvasItemCtrl.setObjectName("canvasItemCtrl")
        self.ctrlLayout = QtGui.QGridLayout(self.canvasItemCtrl)
        self.ctrlLayout.setContentsMargins(0, 0, 0, 0)
        self.ctrlLayout.setSpacing(0)
        self.ctrlLayout.setContentsMargins(0, 0, 0, 0)
        self.ctrlLayout.setObjectName("ctrlLayout")
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "PyQtGraph", None, QtGui.QApplication.UnicodeUTF8))
        self.autoRangeBtn.setText(QtGui.QApplication.translate("Form", "Auto Range", None, QtGui.QApplication.UnicodeUTF8))
        self.redirectCheck.setToolTip(QtGui.QApplication.translate("Form", "Check to display all local items in a remote canvas.", None, QtGui.QApplication.UnicodeUTF8))
        self.redirectCheck.setText(QtGui.QApplication.translate("Form", "Redirect", None, QtGui.QApplication.UnicodeUTF8))
        self.resetTransformsBtn.setText(QtGui.QApplication.translate("Form", "Reset Transforms", None, QtGui.QApplication.UnicodeUTF8))
        self.mirrorSelectionBtn.setText(QtGui.QApplication.translate("Form", "Mirror Selection", None, QtGui.QApplication.UnicodeUTF8))
        self.reflectSelectionBtn.setText(QtGui.QApplication.translate("Form", "MirrorXY", None, QtGui.QApplication.UnicodeUTF8))

from .CanvasManager import CanvasCombo
from ..widgets.TreeWidget import TreeWidget
from ..widgets.GraphicsView import GraphicsView
