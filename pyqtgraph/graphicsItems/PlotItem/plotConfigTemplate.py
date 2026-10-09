from ...Qt import QtCore, QtWidgets

translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(481, 840)
        Form.setWindowTitle(translate("Form", "PyQtGraph"))

        # -- Transform --
        self.transformGroup = QtWidgets.QFrame(Form)
        self.transformGroup.setGeometry(QtCore.QRect(10, 10, 171, 101))
        self.transformGroup.setObjectName("transformGroup")
        self.gridLayout = QtWidgets.QGridLayout(self.transformGroup)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.fftCheck = QtWidgets.QCheckBox(self.transformGroup)
        self.fftCheck.setObjectName("fftCheck")
        self.fftCheck.setText(translate("Form", "Power Spectrum (FFT)"))
        self.gridLayout.addWidget(self.fftCheck, 0, 0, 1, 1)

        self.subtractMeanCheck = QtWidgets.QCheckBox(self.transformGroup)
        self.subtractMeanCheck.setObjectName("subtractMeanCheck")
        self.subtractMeanCheck.setText(translate("Form", "Subtract Mean"))
        self.gridLayout.addWidget(self.subtractMeanCheck, 1, 0, 1, 1)

        self.logXCheck = QtWidgets.QCheckBox(self.transformGroup)
        self.logXCheck.setObjectName("logXCheck")
        self.logXCheck.setText(translate("Form", "Log X"))
        self.gridLayout.addWidget(self.logXCheck, 2, 0, 1, 1)

        self.logYCheck = QtWidgets.QCheckBox(self.transformGroup)
        self.logYCheck.setObjectName("logYCheck")
        self.logYCheck.setText(translate("Form", "Log Y"))
        self.gridLayout.addWidget(self.logYCheck, 3, 0, 1, 1)

        self.derivativeCheck = QtWidgets.QCheckBox(self.transformGroup)
        self.derivativeCheck.setObjectName("derivativeCheck")
        self.derivativeCheck.setText(translate("Form", "dy/dx"))
        self.gridLayout.addWidget(self.derivativeCheck, 4, 0, 1, 1)

        self.phasemapCheck = QtWidgets.QCheckBox(self.transformGroup)
        self.phasemapCheck.setObjectName("phasemapCheck")
        self.phasemapCheck.setText(translate("Form", "Y vs. Y\'"))
        self.gridLayout.addWidget(self.phasemapCheck, 5, 0, 1, 1)

        # -- Downsample / decimate --
        self.decimateGroup = QtWidgets.QFrame(Form)
        self.decimateGroup.setGeometry(QtCore.QRect(10, 140, 191, 171))
        self.decimateGroup.setObjectName("decimateGroup")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.decimateGroup)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName("gridLayout_4")

        self.downsampleCheck = QtWidgets.QCheckBox(self.decimateGroup)
        self.downsampleCheck.setObjectName("downsampleCheck")
        self.downsampleCheck.setText(translate("Form", "Downsample"))
        self.gridLayout_4.addWidget(self.downsampleCheck, 0, 0, 1, 3)

        self.autoDownsampleCheck = QtWidgets.QCheckBox(self.decimateGroup)
        self.autoDownsampleCheck.setChecked(True)
        self.autoDownsampleCheck.setObjectName("autoDownsampleCheck")
        self.autoDownsampleCheck.setToolTip(translate("Form", "Automatically downsample data based on the visible range. This assumes X values are uniformly spaced."))
        self.autoDownsampleCheck.setText(translate("Form", "Auto"))
        self.gridLayout_4.addWidget(self.autoDownsampleCheck, 1, 2, 1, 1)

        self.downsampleSpin = QtWidgets.QSpinBox(self.decimateGroup)
        self.downsampleSpin.setMinimum(1)
        self.downsampleSpin.setMaximum(100000)
        self.downsampleSpin.setProperty("value", 1)
        self.downsampleSpin.setObjectName("downsampleSpin")
        self.downsampleSpin.setToolTip(translate("Form", "Downsample data before plotting. (plot every Nth sample)"))
        self.downsampleSpin.setSuffix(translate("Form", "x"))
        self.gridLayout_4.addWidget(self.downsampleSpin, 1, 1, 1, 1)

        decimateSpacer = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_4.addItem(decimateSpacer, 2, 0, 1, 1)

        self.subsampleRadio = QtWidgets.QRadioButton(self.decimateGroup)
        self.subsampleRadio.setObjectName("subsampleRadio")
        self.subsampleRadio.setToolTip(translate("Form", "Downsample by taking the first of N samples. This method is fastest and least accurate."))
        self.subsampleRadio.setText(translate("Form", "Subsample"))
        self.gridLayout_4.addWidget(self.subsampleRadio, 2, 1, 1, 2)

        self.meanRadio = QtWidgets.QRadioButton(self.decimateGroup)
        self.meanRadio.setObjectName("meanRadio")
        self.meanRadio.setToolTip(translate("Form", "Downsample by taking the mean of N samples."))
        self.meanRadio.setText(translate("Form", "Mean"))
        self.gridLayout_4.addWidget(self.meanRadio, 3, 1, 1, 2)

        self.peakRadio = QtWidgets.QRadioButton(self.decimateGroup)
        self.peakRadio.setChecked(True)
        self.peakRadio.setObjectName("peakRadio")
        self.peakRadio.setToolTip(translate("Form", "Downsample by drawing a saw wave that follows the min and max of the original data. This method produces the best visual representation of the data but is slower."))
        self.peakRadio.setText(translate("Form", "Peak"))
        self.gridLayout_4.addWidget(self.peakRadio, 6, 1, 1, 2)

        self.clipToViewCheck = QtWidgets.QCheckBox(self.decimateGroup)
        self.clipToViewCheck.setObjectName("clipToViewCheck")
        self.clipToViewCheck.setToolTip(translate("Form", "Plot only the portion of each curve that is visible. This assumes X values are uniformly spaced."))
        self.clipToViewCheck.setText(translate("Form", "Clip to View"))
        self.gridLayout_4.addWidget(self.clipToViewCheck, 7, 0, 1, 3)

        self.maxTracesCheck = QtWidgets.QCheckBox(self.decimateGroup)
        self.maxTracesCheck.setObjectName("maxTracesCheck")
        self.maxTracesCheck.setToolTip(translate("Form", "If multiple curves are displayed in this plot, check this box to limit the number of traces that are displayed."))
        self.maxTracesCheck.setText(translate("Form", "Max Traces:"))
        self.gridLayout_4.addWidget(self.maxTracesCheck, 8, 0, 1, 2)

        self.maxTracesSpin = QtWidgets.QSpinBox(self.decimateGroup)
        self.maxTracesSpin.setObjectName("maxTracesSpin")
        self.maxTracesSpin.setToolTip(translate("Form", "If multiple curves are displayed in this plot, check \"Max Traces\" and set this value to limit the number of traces that are displayed."))
        self.gridLayout_4.addWidget(self.maxTracesSpin, 8, 2, 1, 1)

        self.forgetTracesCheck = QtWidgets.QCheckBox(self.decimateGroup)
        self.forgetTracesCheck.setObjectName("forgetTracesCheck")
        self.forgetTracesCheck.setToolTip(translate("Form", "If MaxTraces is checked, remove curves from memory after they are hidden (saves memory, but traces can not be un-hidden)."))
        self.forgetTracesCheck.setText(translate("Form", "Forget hidden traces"))
        self.gridLayout_4.addWidget(self.forgetTracesCheck, 9, 0, 1, 3)

        # -- Alpha --
        self.alphaGroup = QtWidgets.QGroupBox(Form)
        self.alphaGroup.setGeometry(QtCore.QRect(10, 390, 234, 60))
        self.alphaGroup.setCheckable(True)
        self.alphaGroup.setObjectName("alphaGroup")
        self.alphaGroup.setTitle(translate("Form", "Alpha"))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.alphaGroup)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.autoAlphaCheck = QtWidgets.QCheckBox(self.alphaGroup)
        self.autoAlphaCheck.setChecked(False)
        self.autoAlphaCheck.setObjectName("autoAlphaCheck")
        self.autoAlphaCheck.setText(translate("Form", "Auto"))
        self.horizontalLayout.addWidget(self.autoAlphaCheck)

        self.alphaSlider = QtWidgets.QSlider(self.alphaGroup)
        self.alphaSlider.setMaximum(1000)
        self.alphaSlider.setProperty("value", 1000)
        self.alphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.alphaSlider.setObjectName("alphaSlider")
        self.horizontalLayout.addWidget(self.alphaSlider)

        # -- Grid --
        self.gridGroup = QtWidgets.QFrame(Form)
        self.gridGroup.setGeometry(QtCore.QRect(10, 460, 221, 81))
        self.gridGroup.setObjectName("gridGroup")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridGroup)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.xGridCheck = QtWidgets.QCheckBox(self.gridGroup)
        self.xGridCheck.setObjectName("xGridCheck")
        self.xGridCheck.setText(translate("Form", "Show X Grid"))
        self.gridLayout_2.addWidget(self.xGridCheck, 0, 0, 1, 2)

        self.yGridCheck = QtWidgets.QCheckBox(self.gridGroup)
        self.yGridCheck.setObjectName("yGridCheck")
        self.yGridCheck.setText(translate("Form", "Show Y Grid"))
        self.gridLayout_2.addWidget(self.yGridCheck, 1, 0, 1, 2)

        self.label = QtWidgets.QLabel(self.gridGroup)
        self.label.setObjectName("label")
        self.label.setText(translate("Form", "Opacity"))
        self.gridLayout_2.addWidget(self.label, 2, 0, 1, 1)

        self.gridAlphaSlider = QtWidgets.QSlider(self.gridGroup)
        self.gridAlphaSlider.setMaximum(255)
        self.gridAlphaSlider.setProperty("value", 128)
        self.gridAlphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.gridAlphaSlider.setObjectName("gridAlphaSlider")
        self.gridLayout_2.addWidget(self.gridAlphaSlider, 2, 1, 1, 1)

        # -- Points --
        self.pointsGroup = QtWidgets.QGroupBox(Form)
        self.pointsGroup.setGeometry(QtCore.QRect(10, 550, 234, 58))
        self.pointsGroup.setCheckable(True)
        self.pointsGroup.setObjectName("pointsGroup")
        self.pointsGroup.setTitle(translate("Form", "Points"))
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.pointsGroup)
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        self.autoPointsCheck = QtWidgets.QCheckBox(self.pointsGroup)
        self.autoPointsCheck.setChecked(True)
        self.autoPointsCheck.setObjectName("autoPointsCheck")
        self.autoPointsCheck.setText(translate("Form", "Auto"))
        self.verticalLayout_5.addWidget(self.autoPointsCheck)

        # -- Average --
        self.averageGroup = QtWidgets.QGroupBox(Form)
        self.averageGroup.setGeometry(QtCore.QRect(0, 640, 242, 182))
        self.averageGroup.setCheckable(True)
        self.averageGroup.setChecked(False)
        self.averageGroup.setObjectName("averageGroup")
        self.averageGroup.setToolTip(translate("Form", "Display averages of the curves displayed in this plot. The parameter list allows you to choose parameters to average over (if any are available)."))
        self.averageGroup.setTitle(translate("Form", "Average"))
        self.gridLayout_5 = QtWidgets.QGridLayout(self.averageGroup)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setSpacing(0)
        self.gridLayout_5.setObjectName("gridLayout_5")

        self.avgParamList = QtWidgets.QListWidget(self.averageGroup)
        self.avgParamList.setObjectName("avgParamList")
        self.gridLayout_5.addWidget(self.avgParamList, 0, 0, 1, 1)

        QtCore.QMetaObject.connectSlotsByName(Form)
