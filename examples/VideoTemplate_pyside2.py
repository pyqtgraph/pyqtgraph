# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'VideoTemplate.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from pyqtgraph import GraphicsView
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from pyqtgraph import GradientWidget
from pyqtgraph import SpinBox


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(695, 798)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.cudaCheck = QCheckBox(self.centralwidget)
        self.cudaCheck.setObjectName(u"cudaCheck")

        self.gridLayout_2.addWidget(self.cudaCheck, 9, 0, 1, 2)

        self.downsampleCheck = QCheckBox(self.centralwidget)
        self.downsampleCheck.setObjectName(u"downsampleCheck")

        self.gridLayout_2.addWidget(self.downsampleCheck, 8, 0, 1, 2)

        self.scaleCheck = QCheckBox(self.centralwidget)
        self.scaleCheck.setObjectName(u"scaleCheck")

        self.gridLayout_2.addWidget(self.scaleCheck, 4, 0, 1, 1)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.rawRadio = QRadioButton(self.centralwidget)
        self.rawRadio.setObjectName(u"rawRadio")

        self.gridLayout.addWidget(self.rawRadio, 3, 0, 1, 1)

        self.gfxRadio = QRadioButton(self.centralwidget)
        self.gfxRadio.setObjectName(u"gfxRadio")
        self.gfxRadio.setChecked(True)

        self.gridLayout.addWidget(self.gfxRadio, 2, 0, 1, 1)

        self.stack = QStackedWidget(self.centralwidget)
        self.stack.setObjectName(u"stack")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.gridLayout_3 = QGridLayout(self.page)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.graphicsView = GraphicsView(self.page)
        self.graphicsView.setObjectName(u"graphicsView")

        self.gridLayout_3.addWidget(self.graphicsView, 0, 0, 1, 1)

        self.stack.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.gridLayout_4 = QGridLayout(self.page_2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.rawImg = RawImageWidget(self.page_2)
        self.rawImg.setObjectName(u"rawImg")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rawImg.sizePolicy().hasHeightForWidth())
        self.rawImg.setSizePolicy(sizePolicy)

        self.gridLayout_4.addWidget(self.rawImg, 0, 0, 1, 1)

        self.stack.addWidget(self.page_2)

        self.gridLayout.addWidget(self.stack, 0, 0, 1, 1)

        self.rawGLRadio = QRadioButton(self.centralwidget)
        self.rawGLRadio.setObjectName(u"rawGLRadio")

        self.gridLayout.addWidget(self.rawGLRadio, 4, 0, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 4)

        self.dtypeCombo = QComboBox(self.centralwidget)
        self.dtypeCombo.addItem("")
        self.dtypeCombo.addItem("")
        self.dtypeCombo.addItem("")
        self.dtypeCombo.setObjectName(u"dtypeCombo")

        self.gridLayout_2.addWidget(self.dtypeCombo, 3, 2, 1, 1)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 3, 0, 1, 1)

        self.rgbLevelsCheck = QCheckBox(self.centralwidget)
        self.rgbLevelsCheck.setObjectName(u"rgbLevelsCheck")

        self.gridLayout_2.addWidget(self.rgbLevelsCheck, 4, 1, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.minSpin2 = SpinBox(self.centralwidget)
        self.minSpin2.setObjectName(u"minSpin2")
        self.minSpin2.setEnabled(False)

        self.horizontalLayout_2.addWidget(self.minSpin2)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.label_3)

        self.maxSpin2 = SpinBox(self.centralwidget)
        self.maxSpin2.setObjectName(u"maxSpin2")
        self.maxSpin2.setEnabled(False)

        self.horizontalLayout_2.addWidget(self.maxSpin2)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 5, 2, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.minSpin1 = SpinBox(self.centralwidget)
        self.minSpin1.setObjectName(u"minSpin1")

        self.horizontalLayout.addWidget(self.minSpin1)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_2)

        self.maxSpin1 = SpinBox(self.centralwidget)
        self.maxSpin1.setObjectName(u"maxSpin1")

        self.horizontalLayout.addWidget(self.maxSpin1)


        self.gridLayout_2.addLayout(self.horizontalLayout, 4, 2, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.minSpin3 = SpinBox(self.centralwidget)
        self.minSpin3.setObjectName(u"minSpin3")
        self.minSpin3.setEnabled(False)

        self.horizontalLayout_3.addWidget(self.minSpin3)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_4)

        self.maxSpin3 = SpinBox(self.centralwidget)
        self.maxSpin3.setObjectName(u"maxSpin3")
        self.maxSpin3.setEnabled(False)

        self.horizontalLayout_3.addWidget(self.maxSpin3)


        self.gridLayout_2.addLayout(self.horizontalLayout_3, 6, 2, 1, 1)

        self.lutCheck = QCheckBox(self.centralwidget)
        self.lutCheck.setObjectName(u"lutCheck")

        self.gridLayout_2.addWidget(self.lutCheck, 7, 0, 1, 1)

        self.alphaCheck = QCheckBox(self.centralwidget)
        self.alphaCheck.setObjectName(u"alphaCheck")

        self.gridLayout_2.addWidget(self.alphaCheck, 7, 1, 1, 1)

        self.gradient = GradientWidget(self.centralwidget)
        self.gradient.setObjectName(u"gradient")
        sizePolicy.setHeightForWidth(self.gradient.sizePolicy().hasHeightForWidth())
        self.gradient.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.gradient, 7, 2, 1, 2)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 3, 3, 1, 1)

        self.fpsLabel = QLabel(self.centralwidget)
        self.fpsLabel.setObjectName(u"fpsLabel")
        font = QFont()
        font.setPointSize(12)
        self.fpsLabel.setFont(font)
        self.fpsLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.fpsLabel, 0, 0, 1, 4)

        self.rgbCheck = QCheckBox(self.centralwidget)
        self.rgbCheck.setObjectName(u"rgbCheck")

        self.gridLayout_2.addWidget(self.rgbCheck, 3, 1, 1, 1)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_2.addWidget(self.label_5, 2, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.framesSpin = QSpinBox(self.centralwidget)
        self.framesSpin.setObjectName(u"framesSpin")
        self.framesSpin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.framesSpin.setValue(10)

        self.horizontalLayout_4.addWidget(self.framesSpin)

        self.widthSpin = QSpinBox(self.centralwidget)
        self.widthSpin.setObjectName(u"widthSpin")
        self.widthSpin.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.widthSpin.setMaximum(10000)
        self.widthSpin.setValue(512)

        self.horizontalLayout_4.addWidget(self.widthSpin)

        self.heightSpin = QSpinBox(self.centralwidget)
        self.heightSpin.setObjectName(u"heightSpin")
        self.heightSpin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.heightSpin.setMaximum(10000)
        self.heightSpin.setValue(512)

        self.horizontalLayout_4.addWidget(self.heightSpin)


        self.gridLayout_2.addLayout(self.horizontalLayout_4, 2, 1, 1, 2)

        self.sizeLabel = QLabel(self.centralwidget)
        self.sizeLabel.setObjectName(u"sizeLabel")

        self.gridLayout_2.addWidget(self.sizeLabel, 2, 3, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stack.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.cudaCheck.setText(QCoreApplication.translate("MainWindow", u"Use CUDA (GPU) if available", None))
        self.downsampleCheck.setText(QCoreApplication.translate("MainWindow", u"Auto downsample", None))
        self.scaleCheck.setText(QCoreApplication.translate("MainWindow", u"Scale Data", None))
        self.rawRadio.setText(QCoreApplication.translate("MainWindow", u"RawImageWidget", None))
        self.gfxRadio.setText(QCoreApplication.translate("MainWindow", u"GraphicsView + ImageItem", None))
        self.rawGLRadio.setText(QCoreApplication.translate("MainWindow", u"RawGLImageWidget", None))
        self.dtypeCombo.setItemText(0, QCoreApplication.translate("MainWindow", u"uint8", None))
        self.dtypeCombo.setItemText(1, QCoreApplication.translate("MainWindow", u"uint16", None))
        self.dtypeCombo.setItemText(2, QCoreApplication.translate("MainWindow", u"float", None))

        self.label.setText(QCoreApplication.translate("MainWindow", u"Data type", None))
        self.rgbLevelsCheck.setText(QCoreApplication.translate("MainWindow", u"RGB", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"<--->", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"<--->", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"<--->", None))
        self.lutCheck.setText(QCoreApplication.translate("MainWindow", u"Use Lookup  Table", None))
        self.alphaCheck.setText(QCoreApplication.translate("MainWindow", u"alpha", None))
        self.fpsLabel.setText(QCoreApplication.translate("MainWindow", u"FPS", None))
        self.rgbCheck.setText(QCoreApplication.translate("MainWindow", u"RGB", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Image size", None))
        self.sizeLabel.setText("")
    # retranslateUi

