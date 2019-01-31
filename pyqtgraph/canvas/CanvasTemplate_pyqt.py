# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/canvas/CanvasTemplate.ui'
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
        Form.resize(821, 578)
        self.gridLayout_2 = QtGui.QGridLayout(Form)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.view = GraphicsView(self.splitter)
        self.view.setObjectName(_fromUtf8("view"))
        self.vsplitter = QtGui.QSplitter(self.splitter)
        self.vsplitter.setOrientation(QtCore.Qt.Vertical)
        self.vsplitter.setObjectName(_fromUtf8("vsplitter"))
        self.canvasCtrlWidget = QtGui.QWidget(self.vsplitter)
        self.canvasCtrlWidget.setObjectName(_fromUtf8("canvasCtrlWidget"))
        self.gridLayout = QtGui.QGridLayout(self.canvasCtrlWidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.autoRangeBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.autoRangeBtn.sizePolicy().hasHeightForWidth())
        self.autoRangeBtn.setSizePolicy(sizePolicy)
        self.autoRangeBtn.setObjectName(_fromUtf8("autoRangeBtn"))
        self.gridLayout.addWidget(self.autoRangeBtn, 0, 0, 1, 2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.redirectCheck = QtGui.QCheckBox(self.canvasCtrlWidget)
        self.redirectCheck.setObjectName(_fromUtf8("redirectCheck"))
        self.horizontalLayout.addWidget(self.redirectCheck)
        self.redirectCombo = CanvasCombo(self.canvasCtrlWidget)
        self.redirectCombo.setObjectName(_fromUtf8("redirectCombo"))
        self.horizontalLayout.addWidget(self.redirectCombo)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        self.itemList = TreeWidget(self.canvasCtrlWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.itemList.sizePolicy().hasHeightForWidth())
        self.itemList.setSizePolicy(sizePolicy)
        self.itemList.setHeaderHidden(True)
        self.itemList.setObjectName(_fromUtf8("itemList"))
        self.itemList.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout.addWidget(self.itemList, 2, 0, 1, 2)
        self.resetTransformsBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        self.resetTransformsBtn.setObjectName(_fromUtf8("resetTransformsBtn"))
        self.gridLayout.addWidget(self.resetTransformsBtn, 3, 0, 1, 2)
        self.mirrorSelectionBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        self.mirrorSelectionBtn.setObjectName(_fromUtf8("mirrorSelectionBtn"))
        self.gridLayout.addWidget(self.mirrorSelectionBtn, 4, 0, 1, 1)
        self.reflectSelectionBtn = QtGui.QPushButton(self.canvasCtrlWidget)
        self.reflectSelectionBtn.setObjectName(_fromUtf8("reflectSelectionBtn"))
        self.gridLayout.addWidget(self.reflectSelectionBtn, 4, 1, 1, 1)
        self.canvasItemCtrl = QtGui.QWidget(self.vsplitter)
        self.canvasItemCtrl.setObjectName(_fromUtf8("canvasItemCtrl"))
        self.ctrlLayout = QtGui.QGridLayout(self.canvasItemCtrl)
        self.ctrlLayout.setMargin(0)
        self.ctrlLayout.setSpacing(0)
        self.ctrlLayout.setObjectName(_fromUtf8("ctrlLayout"))
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.autoRangeBtn.setText(_translate("Form", "Auto Range", None))
        self.redirectCheck.setToolTip(_translate("Form", "Check to display all local items in a remote canvas.", None))
        self.redirectCheck.setText(_translate("Form", "Redirect", None))
        self.resetTransformsBtn.setText(_translate("Form", "Reset Transforms", None))
        self.mirrorSelectionBtn.setText(_translate("Form", "Mirror Selection", None))
        self.reflectSelectionBtn.setText(_translate("Form", "MirrorXY", None))

from ..widgets.GraphicsView import GraphicsView
from ..widgets.TreeWidget import TreeWidget
from .CanvasManager import CanvasCombo
