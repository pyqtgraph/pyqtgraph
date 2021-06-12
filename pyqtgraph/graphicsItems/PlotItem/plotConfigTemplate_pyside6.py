# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plotConfigTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(481, 840)
        self.averageGroup = QGroupBox(Form)
        self.averageGroup.setObjectName(u"averageGroup")
        self.averageGroup.setGeometry(QRect(0, 640, 242, 182))
        self.averageGroup.setCheckable(True)
        self.averageGroup.setChecked(False)
        self.gridLayout_5 = QGridLayout(self.averageGroup)
        self.gridLayout_5.setSpacing(0)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.avgParamList = QListWidget(self.averageGroup)
        self.avgParamList.setObjectName(u"avgParamList")

        self.gridLayout_5.addWidget(self.avgParamList, 0, 0, 1, 1)

        self.decimateGroup = QFrame(Form)
        self.decimateGroup.setObjectName(u"decimateGroup")
        self.decimateGroup.setGeometry(QRect(10, 140, 191, 171))
        self.gridLayout_4 = QGridLayout(self.decimateGroup)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.clipToViewCheck = QCheckBox(self.decimateGroup)
        self.clipToViewCheck.setObjectName(u"clipToViewCheck")

        self.gridLayout_4.addWidget(self.clipToViewCheck, 7, 0, 1, 3)

        self.maxTracesCheck = QCheckBox(self.decimateGroup)
        self.maxTracesCheck.setObjectName(u"maxTracesCheck")

        self.gridLayout_4.addWidget(self.maxTracesCheck, 8, 0, 1, 2)

        self.downsampleCheck = QCheckBox(self.decimateGroup)
        self.downsampleCheck.setObjectName(u"downsampleCheck")

        self.gridLayout_4.addWidget(self.downsampleCheck, 0, 0, 1, 3)

        self.peakRadio = QRadioButton(self.decimateGroup)
        self.peakRadio.setObjectName(u"peakRadio")
        self.peakRadio.setChecked(True)

        self.gridLayout_4.addWidget(self.peakRadio, 6, 1, 1, 2)

        self.maxTracesSpin = QSpinBox(self.decimateGroup)
        self.maxTracesSpin.setObjectName(u"maxTracesSpin")

        self.gridLayout_4.addWidget(self.maxTracesSpin, 8, 2, 1, 1)

        self.forgetTracesCheck = QCheckBox(self.decimateGroup)
        self.forgetTracesCheck.setObjectName(u"forgetTracesCheck")

        self.gridLayout_4.addWidget(self.forgetTracesCheck, 9, 0, 1, 3)

        self.meanRadio = QRadioButton(self.decimateGroup)
        self.meanRadio.setObjectName(u"meanRadio")

        self.gridLayout_4.addWidget(self.meanRadio, 3, 1, 1, 2)

        self.subsampleRadio = QRadioButton(self.decimateGroup)
        self.subsampleRadio.setObjectName(u"subsampleRadio")

        self.gridLayout_4.addWidget(self.subsampleRadio, 2, 1, 1, 2)

        self.autoDownsampleCheck = QCheckBox(self.decimateGroup)
        self.autoDownsampleCheck.setObjectName(u"autoDownsampleCheck")
        self.autoDownsampleCheck.setChecked(True)

        self.gridLayout_4.addWidget(self.autoDownsampleCheck, 1, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(30, 20, QSizePolicy.Maximum, QSizePolicy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer, 2, 0, 1, 1)

        self.downsampleSpin = QSpinBox(self.decimateGroup)
        self.downsampleSpin.setObjectName(u"downsampleSpin")
        self.downsampleSpin.setMinimum(1)
        self.downsampleSpin.setMaximum(100000)
        self.downsampleSpin.setValue(1)

        self.gridLayout_4.addWidget(self.downsampleSpin, 1, 1, 1, 1)

        self.transformGroup = QFrame(Form)
        self.transformGroup.setObjectName(u"transformGroup")
        self.transformGroup.setGeometry(QRect(10, 10, 171, 101))
        self.gridLayout = QGridLayout(self.transformGroup)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.logYCheck = QCheckBox(self.transformGroup)
        self.logYCheck.setObjectName(u"logYCheck")

        self.gridLayout.addWidget(self.logYCheck, 2, 0, 1, 1)

        self.logXCheck = QCheckBox(self.transformGroup)
        self.logXCheck.setObjectName(u"logXCheck")

        self.gridLayout.addWidget(self.logXCheck, 1, 0, 1, 1)

        self.fftCheck = QCheckBox(self.transformGroup)
        self.fftCheck.setObjectName(u"fftCheck")

        self.gridLayout.addWidget(self.fftCheck, 0, 0, 1, 1)

        self.derivativeCheck = QCheckBox(self.transformGroup)
        self.derivativeCheck.setObjectName(u"derivativeCheck")

        self.gridLayout.addWidget(self.derivativeCheck, 3, 0, 1, 1)

        self.phasemapCheck = QCheckBox(self.transformGroup)
        self.phasemapCheck.setObjectName(u"phasemapCheck")

        self.gridLayout.addWidget(self.phasemapCheck, 4, 0, 1, 1)

        self.pointsGroup = QGroupBox(Form)
        self.pointsGroup.setObjectName(u"pointsGroup")
        self.pointsGroup.setGeometry(QRect(10, 550, 234, 58))
        self.pointsGroup.setCheckable(True)
        self.verticalLayout_5 = QVBoxLayout(self.pointsGroup)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.autoPointsCheck = QCheckBox(self.pointsGroup)
        self.autoPointsCheck.setObjectName(u"autoPointsCheck")
        self.autoPointsCheck.setChecked(True)

        self.verticalLayout_5.addWidget(self.autoPointsCheck)

        self.gridGroup = QFrame(Form)
        self.gridGroup.setObjectName(u"gridGroup")
        self.gridGroup.setGeometry(QRect(10, 460, 221, 81))
        self.gridLayout_2 = QGridLayout(self.gridGroup)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.xGridCheck = QCheckBox(self.gridGroup)
        self.xGridCheck.setObjectName(u"xGridCheck")

        self.gridLayout_2.addWidget(self.xGridCheck, 0, 0, 1, 2)

        self.yGridCheck = QCheckBox(self.gridGroup)
        self.yGridCheck.setObjectName(u"yGridCheck")

        self.gridLayout_2.addWidget(self.yGridCheck, 1, 0, 1, 2)

        self.gridAlphaSlider = QSlider(self.gridGroup)
        self.gridAlphaSlider.setObjectName(u"gridAlphaSlider")
        self.gridAlphaSlider.setMaximum(255)
        self.gridAlphaSlider.setValue(128)
        self.gridAlphaSlider.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.gridAlphaSlider, 2, 1, 1, 1)

        self.label = QLabel(self.gridGroup)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 2, 0, 1, 1)

        self.alphaGroup = QGroupBox(Form)
        self.alphaGroup.setObjectName(u"alphaGroup")
        self.alphaGroup.setGeometry(QRect(10, 390, 234, 60))
        self.alphaGroup.setCheckable(True)
        self.horizontalLayout = QHBoxLayout(self.alphaGroup)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.autoAlphaCheck = QCheckBox(self.alphaGroup)
        self.autoAlphaCheck.setObjectName(u"autoAlphaCheck")
        self.autoAlphaCheck.setChecked(False)

        self.horizontalLayout.addWidget(self.autoAlphaCheck)

        self.alphaSlider = QSlider(self.alphaGroup)
        self.alphaSlider.setObjectName(u"alphaSlider")
        self.alphaSlider.setMaximum(1000)
        self.alphaSlider.setValue(1000)
        self.alphaSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout.addWidget(self.alphaSlider)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
#if QT_CONFIG(tooltip)
        self.averageGroup.setToolTip(QCoreApplication.translate("Form", u"Display averages of the curves displayed in this plot. The parameter list allows you to choose parameters to average over (if any are available).", None))
#endif // QT_CONFIG(tooltip)
        self.averageGroup.setTitle(QCoreApplication.translate("Form", u"Average", None))
#if QT_CONFIG(tooltip)
        self.clipToViewCheck.setToolTip(QCoreApplication.translate("Form", u"Plot only the portion of each curve that is visible. This assumes X values are uniformly spaced.", None))
#endif // QT_CONFIG(tooltip)
        self.clipToViewCheck.setText(QCoreApplication.translate("Form", u"Clip to View", None))
#if QT_CONFIG(tooltip)
        self.maxTracesCheck.setToolTip(QCoreApplication.translate("Form", u"If multiple curves are displayed in this plot, check this box to limit the number of traces that are displayed.", None))
#endif // QT_CONFIG(tooltip)
        self.maxTracesCheck.setText(QCoreApplication.translate("Form", u"Max Traces:", None))
        self.downsampleCheck.setText(QCoreApplication.translate("Form", u"Downsample", None))
#if QT_CONFIG(tooltip)
        self.peakRadio.setToolTip(QCoreApplication.translate("Form", u"Downsample by drawing a saw wave that follows the min and max of the original data. This method produces the best visual representation of the data but is slower.", None))
#endif // QT_CONFIG(tooltip)
        self.peakRadio.setText(QCoreApplication.translate("Form", u"Peak", None))
#if QT_CONFIG(tooltip)
        self.maxTracesSpin.setToolTip(QCoreApplication.translate("Form", u"If multiple curves are displayed in this plot, check \"Max Traces\" and set this value to limit the number of traces that are displayed.", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.forgetTracesCheck.setToolTip(QCoreApplication.translate("Form", u"If MaxTraces is checked, remove curves from memory after they are hidden (saves memory, but traces can not be un-hidden).", None))
#endif // QT_CONFIG(tooltip)
        self.forgetTracesCheck.setText(QCoreApplication.translate("Form", u"Forget hidden traces", None))
#if QT_CONFIG(tooltip)
        self.meanRadio.setToolTip(QCoreApplication.translate("Form", u"Downsample by taking the mean of N samples.", None))
#endif // QT_CONFIG(tooltip)
        self.meanRadio.setText(QCoreApplication.translate("Form", u"Mean", None))
#if QT_CONFIG(tooltip)
        self.subsampleRadio.setToolTip(QCoreApplication.translate("Form", u"Downsample by taking the first of N samples. This method is fastest and least accurate.", None))
#endif // QT_CONFIG(tooltip)
        self.subsampleRadio.setText(QCoreApplication.translate("Form", u"Subsample", None))
#if QT_CONFIG(tooltip)
        self.autoDownsampleCheck.setToolTip(QCoreApplication.translate("Form", u"Automatically downsample data based on the visible range. This assumes X values are uniformly spaced.", None))
#endif // QT_CONFIG(tooltip)
        self.autoDownsampleCheck.setText(QCoreApplication.translate("Form", u"Auto", None))
#if QT_CONFIG(tooltip)
        self.downsampleSpin.setToolTip(QCoreApplication.translate("Form", u"Downsample data before plotting. (plot every Nth sample)", None))
#endif // QT_CONFIG(tooltip)
        self.downsampleSpin.setSuffix(QCoreApplication.translate("Form", u"x", None))
        self.logYCheck.setText(QCoreApplication.translate("Form", u"Log Y", None))
        self.logXCheck.setText(QCoreApplication.translate("Form", u"Log X", None))
        self.fftCheck.setText(QCoreApplication.translate("Form", u"Power Spectrum (FFT)", None))
        self.derivativeCheck.setText(QCoreApplication.translate("Form", u"dy/dx", None))
        self.phasemapCheck.setText(QCoreApplication.translate("Form", u"Y vs. Y'", None))
        self.pointsGroup.setTitle(QCoreApplication.translate("Form", u"Points", None))
        self.autoPointsCheck.setText(QCoreApplication.translate("Form", u"Auto", None))
        self.xGridCheck.setText(QCoreApplication.translate("Form", u"Show X Grid", None))
        self.yGridCheck.setText(QCoreApplication.translate("Form", u"Show Y Grid", None))
        self.label.setText(QCoreApplication.translate("Form", u"Opacity", None))
        self.alphaGroup.setTitle(QCoreApplication.translate("Form", u"Alpha", None))
        self.autoAlphaCheck.setText(QCoreApplication.translate("Form", u"Auto", None))
    # retranslateUi

