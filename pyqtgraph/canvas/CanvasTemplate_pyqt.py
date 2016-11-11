# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyqtgraph/canvas/CanvasTemplate.ui'
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
        Form.resize(490, 414)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.view = GraphicsView(self.splitter)
        self.view.setObjectName(_fromUtf8("view"))
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.autoRangeBtn = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.autoRangeBtn.sizePolicy().hasHeightForWidth())
        self.autoRangeBtn.setSizePolicy(sizePolicy)
        self.autoRangeBtn.setObjectName(_fromUtf8("autoRangeBtn"))
        self.gridLayout_2.addWidget(self.autoRangeBtn, 2, 0, 1, 2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.redirectCheck = QtGui.QCheckBox(self.layoutWidget)
        self.redirectCheck.setObjectName(_fromUtf8("redirectCheck"))
        self.horizontalLayout.addWidget(self.redirectCheck)
        self.redirectCombo = CanvasCombo(self.layoutWidget)
        self.redirectCombo.setObjectName(_fromUtf8("redirectCombo"))
        self.horizontalLayout.addWidget(self.redirectCombo)
        self.gridLayout_2.addLayout(self.horizontalLayout, 5, 0, 1, 2)
        self.itemList = TreeWidget(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.itemList.sizePolicy().hasHeightForWidth())
        self.itemList.setSizePolicy(sizePolicy)
        self.itemList.setHeaderHidden(True)
        self.itemList.setObjectName(_fromUtf8("itemList"))
        self.itemList.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout_2.addWidget(self.itemList, 6, 0, 1, 2)
        self.ctrlLayout = QtGui.QGridLayout()
        self.ctrlLayout.setSpacing(0)
        self.ctrlLayout.setObjectName(_fromUtf8("ctrlLayout"))
        self.gridLayout_2.addLayout(self.ctrlLayout, 10, 0, 1, 2)
        self.resetTransformsBtn = QtGui.QPushButton(self.layoutWidget)
        self.resetTransformsBtn.setObjectName(_fromUtf8("resetTransformsBtn"))
        self.gridLayout_2.addWidget(self.resetTransformsBtn, 7, 0, 1, 1)
        self.mirrorSelectionBtn = QtGui.QPushButton(self.layoutWidget)
        self.mirrorSelectionBtn.setObjectName(_fromUtf8("mirrorSelectionBtn"))
        self.gridLayout_2.addWidget(self.mirrorSelectionBtn, 3, 0, 1, 1)
        self.reflectSelectionBtn = QtGui.QPushButton(self.layoutWidget)
        self.reflectSelectionBtn.setObjectName(_fromUtf8("reflectSelectionBtn"))
        self.gridLayout_2.addWidget(self.reflectSelectionBtn, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

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
