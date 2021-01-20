# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ImageViewTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.0.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ..widgets.PlotWidget import PlotWidget
from ..widgets.GraphicsView import GraphicsView
from ..widgets.HistogramLUTWidget import HistogramLUTWidget


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(726, 588)
        self.gridLayout_3 = QGridLayout(Form)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.gridLayout = QGridLayout(self.layoutWidget)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.graphicsView = GraphicsView(self.layoutWidget)
        self.graphicsView.setObjectName(u"graphicsView")

        self.gridLayout.addWidget(self.graphicsView, 0, 0, 2, 1)

        self.histogram = HistogramLUTWidget(self.layoutWidget)
        self.histogram.setObjectName(u"histogram")

        self.gridLayout.addWidget(self.histogram, 0, 1, 1, 2)

        self.roiBtn = QPushButton(self.layoutWidget)
        self.roiBtn.setObjectName(u"roiBtn")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.roiBtn.sizePolicy().hasHeightForWidth())
        self.roiBtn.setSizePolicy(sizePolicy)
        self.roiBtn.setCheckable(True)

        self.gridLayout.addWidget(self.roiBtn, 1, 1, 1, 1)

        self.menuBtn = QPushButton(self.layoutWidget)
        self.menuBtn.setObjectName(u"menuBtn")
        sizePolicy.setHeightForWidth(self.menuBtn.sizePolicy().hasHeightForWidth())
        self.menuBtn.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.menuBtn, 1, 2, 1, 1)

        self.splitter.addWidget(self.layoutWidget)
        self.roiPlot = PlotWidget(self.splitter)
        self.roiPlot.setObjectName(u"roiPlot")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.roiPlot.sizePolicy().hasHeightForWidth())
        self.roiPlot.setSizePolicy(sizePolicy1)
        self.roiPlot.setMinimumSize(QSize(0, 40))
        self.splitter.addWidget(self.roiPlot)

        self.gridLayout_3.addWidget(self.splitter, 0, 0, 1, 1)

        self.normGroup = QGroupBox(Form)
        self.normGroup.setObjectName(u"normGroup")
        self.gridLayout_2 = QGridLayout(self.normGroup)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.normSubtractRadio = QRadioButton(self.normGroup)
        self.normSubtractRadio.setObjectName(u"normSubtractRadio")

        self.gridLayout_2.addWidget(self.normSubtractRadio, 0, 2, 1, 1)

        self.normDivideRadio = QRadioButton(self.normGroup)
        self.normDivideRadio.setObjectName(u"normDivideRadio")
        self.normDivideRadio.setChecked(False)

        self.gridLayout_2.addWidget(self.normDivideRadio, 0, 1, 1, 1)

        self.label_5 = QLabel(self.normGroup)
        self.label_5.setObjectName(u"label_5")
        font = QFont()
        font.setBold(True)
        self.label_5.setFont(font)

        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)

        self.label_3 = QLabel(self.normGroup)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)

        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)

        self.label_4 = QLabel(self.normGroup)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)

        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)

        self.normROICheck = QCheckBox(self.normGroup)
        self.normROICheck.setObjectName(u"normROICheck")

        self.gridLayout_2.addWidget(self.normROICheck, 1, 1, 1, 1)

        self.normXBlurSpin = QDoubleSpinBox(self.normGroup)
        self.normXBlurSpin.setObjectName(u"normXBlurSpin")

        self.gridLayout_2.addWidget(self.normXBlurSpin, 2, 2, 1, 1)

        self.label_8 = QLabel(self.normGroup)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_8, 2, 1, 1, 1)

        self.label_9 = QLabel(self.normGroup)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_9, 2, 3, 1, 1)

        self.normYBlurSpin = QDoubleSpinBox(self.normGroup)
        self.normYBlurSpin.setObjectName(u"normYBlurSpin")

        self.gridLayout_2.addWidget(self.normYBlurSpin, 2, 4, 1, 1)

        self.label_10 = QLabel(self.normGroup)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_10, 2, 5, 1, 1)

        self.normOffRadio = QRadioButton(self.normGroup)
        self.normOffRadio.setObjectName(u"normOffRadio")
        self.normOffRadio.setChecked(True)

        self.gridLayout_2.addWidget(self.normOffRadio, 0, 3, 1, 1)

        self.normTimeRangeCheck = QCheckBox(self.normGroup)
        self.normTimeRangeCheck.setObjectName(u"normTimeRangeCheck")

        self.gridLayout_2.addWidget(self.normTimeRangeCheck, 1, 3, 1, 1)

        self.normFrameCheck = QCheckBox(self.normGroup)
        self.normFrameCheck.setObjectName(u"normFrameCheck")

        self.gridLayout_2.addWidget(self.normFrameCheck, 1, 2, 1, 1)

        self.normTBlurSpin = QDoubleSpinBox(self.normGroup)
        self.normTBlurSpin.setObjectName(u"normTBlurSpin")

        self.gridLayout_2.addWidget(self.normTBlurSpin, 2, 6, 1, 1)


        self.gridLayout_3.addWidget(self.normGroup, 1, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.roiBtn.setText(QCoreApplication.translate("Form", u"ROI", None))
        self.menuBtn.setText(QCoreApplication.translate("Form", u"Menu", None))
        self.normGroup.setTitle(QCoreApplication.translate("Form", u"Normalization", None))
        self.normSubtractRadio.setText(QCoreApplication.translate("Form", u"Subtract", None))
        self.normDivideRadio.setText(QCoreApplication.translate("Form", u"Divide", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"Operation:", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Mean:", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"Blur:", None))
        self.normROICheck.setText(QCoreApplication.translate("Form", u"ROI", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"X", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"Y", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"T", None))
        self.normOffRadio.setText(QCoreApplication.translate("Form", u"Off", None))
        self.normTimeRangeCheck.setText(QCoreApplication.translate("Form", u"Time range", None))
        self.normFrameCheck.setText(QCoreApplication.translate("Form", u"Frame", None))
    # retranslateUi

