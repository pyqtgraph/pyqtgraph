
################################################################################
## Form generated from reading UI file 'CanvasTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ..widgets.TreeWidget import TreeWidget
from ..widgets.GraphicsView import GraphicsView
from .CanvasManager import CanvasCombo


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(821, 578)
        self.gridLayout_2 = QGridLayout(Form)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.view = GraphicsView(self.splitter)
        self.view.setObjectName(u"view")
        self.splitter.addWidget(self.view)
        self.vsplitter = QSplitter(self.splitter)
        self.vsplitter.setObjectName(u"vsplitter")
        self.vsplitter.setOrientation(Qt.Vertical)
        self.canvasCtrlWidget = QWidget(self.vsplitter)
        self.canvasCtrlWidget.setObjectName(u"canvasCtrlWidget")
        self.gridLayout = QGridLayout(self.canvasCtrlWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.autoRangeBtn = QPushButton(self.canvasCtrlWidget)
        self.autoRangeBtn.setObjectName(u"autoRangeBtn")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.autoRangeBtn.sizePolicy().hasHeightForWidth())
        self.autoRangeBtn.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.autoRangeBtn, 0, 0, 1, 2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.redirectCheck = QCheckBox(self.canvasCtrlWidget)
        self.redirectCheck.setObjectName(u"redirectCheck")

        self.horizontalLayout.addWidget(self.redirectCheck)

        self.redirectCombo = CanvasCombo(self.canvasCtrlWidget)
        self.redirectCombo.setObjectName(u"redirectCombo")

        self.horizontalLayout.addWidget(self.redirectCombo)


        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 2)

        self.itemList = TreeWidget(self.canvasCtrlWidget)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.itemList.setHeaderItem(__qtreewidgetitem)
        self.itemList.setObjectName(u"itemList")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(100)
        sizePolicy1.setHeightForWidth(self.itemList.sizePolicy().hasHeightForWidth())
        self.itemList.setSizePolicy(sizePolicy1)
        self.itemList.setHeaderHidden(True)

        self.gridLayout.addWidget(self.itemList, 2, 0, 1, 2)

        self.resetTransformsBtn = QPushButton(self.canvasCtrlWidget)
        self.resetTransformsBtn.setObjectName(u"resetTransformsBtn")

        self.gridLayout.addWidget(self.resetTransformsBtn, 3, 0, 1, 2)

        self.mirrorSelectionBtn = QPushButton(self.canvasCtrlWidget)
        self.mirrorSelectionBtn.setObjectName(u"mirrorSelectionBtn")

        self.gridLayout.addWidget(self.mirrorSelectionBtn, 4, 0, 1, 1)

        self.reflectSelectionBtn = QPushButton(self.canvasCtrlWidget)
        self.reflectSelectionBtn.setObjectName(u"reflectSelectionBtn")

        self.gridLayout.addWidget(self.reflectSelectionBtn, 4, 1, 1, 1)

        self.vsplitter.addWidget(self.canvasCtrlWidget)
        self.canvasItemCtrl = QWidget(self.vsplitter)
        self.canvasItemCtrl.setObjectName(u"canvasItemCtrl")
        self.ctrlLayout = QGridLayout(self.canvasItemCtrl)
        self.ctrlLayout.setSpacing(0)
        self.ctrlLayout.setContentsMargins(0, 0, 0, 0)
        self.ctrlLayout.setObjectName(u"ctrlLayout")
        self.vsplitter.addWidget(self.canvasItemCtrl)
        self.splitter.addWidget(self.vsplitter)

        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.autoRangeBtn.setText(QCoreApplication.translate("Form", u"Auto Range", None))
#if QT_CONFIG(tooltip)
        self.redirectCheck.setToolTip(QCoreApplication.translate("Form", u"Check to display all local items in a remote canvas.", None))
#endif // QT_CONFIG(tooltip)
        self.redirectCheck.setText(QCoreApplication.translate("Form", u"Redirect", None))
        self.resetTransformsBtn.setText(QCoreApplication.translate("Form", u"Reset Transforms", None))
        self.mirrorSelectionBtn.setText(QCoreApplication.translate("Form", u"Mirror Selection", None))
        self.reflectSelectionBtn.setText(QCoreApplication.translate("Form", u"MirrorXY", None))
    # retranslateUi
