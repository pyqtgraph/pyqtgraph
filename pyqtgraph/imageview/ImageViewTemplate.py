from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.GraphicsView import GraphicsView
from ..widgets.HistogramLUTWidget import HistogramLUTWidget
from ..widgets.PlotWidget import PlotWidget

translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(726, 588)
        Form.setWindowTitle(translate("Form", "PyQtGraph"))

        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("splitter")

        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.graphicsView = GraphicsView(self.layoutWidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 2, 1)

        self.histogram = HistogramLUTWidget(self.layoutWidget)
        self.histogram.setObjectName("histogram")
        self.gridLayout.addWidget(self.histogram, 0, 1, 1, 2)

        self.roiBtn = QtWidgets.QPushButton(self.layoutWidget)
        roiBtnSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        roiBtnSizePolicy.setHorizontalStretch(0)
        roiBtnSizePolicy.setVerticalStretch(1)
        roiBtnSizePolicy.setHeightForWidth(self.roiBtn.sizePolicy().hasHeightForWidth())
        self.roiBtn.setSizePolicy(roiBtnSizePolicy)
        self.roiBtn.setCheckable(True)
        self.roiBtn.setObjectName("roiBtn")
        self.roiBtn.setText(translate("Form", "ROI"))
        self.gridLayout.addWidget(self.roiBtn, 1, 1, 1, 1)

        self.menuBtn = QtWidgets.QPushButton(self.layoutWidget)
        menuBtnSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        menuBtnSizePolicy.setHorizontalStretch(0)
        menuBtnSizePolicy.setVerticalStretch(1)
        menuBtnSizePolicy.setHeightForWidth(self.menuBtn.sizePolicy().hasHeightForWidth())
        self.menuBtn.setSizePolicy(menuBtnSizePolicy)
        self.menuBtn.setObjectName("menuBtn")
        self.menuBtn.setText(translate("Form", "Menu"))
        self.gridLayout.addWidget(self.menuBtn, 1, 2, 1, 1)

        self.roiPlot = PlotWidget(self.splitter)
        roiPlotSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        roiPlotSizePolicy.setHorizontalStretch(0)
        roiPlotSizePolicy.setVerticalStretch(0)
        roiPlotSizePolicy.setHeightForWidth(self.roiPlot.sizePolicy().hasHeightForWidth())
        self.roiPlot.setSizePolicy(roiPlotSizePolicy)
        self.roiPlot.setMinimumSize(QtCore.QSize(0, 40))
        self.roiPlot.setObjectName("roiPlot")

        self.gridLayout_3.addWidget(self.splitter, 0, 0, 1, 1)

        self.normGroup = QtWidgets.QGroupBox(Form)
        self.normGroup.setObjectName("normGroup")
        self.normGroup.setTitle(translate("Form", "Normalization"))
        self.gridLayout_2 = QtWidgets.QGridLayout(self.normGroup)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")

        boldFont = QtGui.QFont()
        boldFont.setBold(True)

        self.label_5 = QtWidgets.QLabel(self.normGroup)
        self.label_5.setFont(boldFont)
        self.label_5.setObjectName("label_5")
        self.label_5.setText(translate("Form", "Operation:"))
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)

        self.normDivideRadio = QtWidgets.QRadioButton(self.normGroup)
        self.normDivideRadio.setChecked(False)
        self.normDivideRadio.setObjectName("normDivideRadio")
        self.normDivideRadio.setText(translate("Form", "Divide"))
        self.gridLayout_2.addWidget(self.normDivideRadio, 0, 1, 1, 1)

        self.normSubtractRadio = QtWidgets.QRadioButton(self.normGroup)
        self.normSubtractRadio.setObjectName("normSubtractRadio")
        self.normSubtractRadio.setText(translate("Form", "Subtract"))
        self.gridLayout_2.addWidget(self.normSubtractRadio, 0, 2, 1, 1)

        self.normOffRadio = QtWidgets.QRadioButton(self.normGroup)
        self.normOffRadio.setChecked(True)
        self.normOffRadio.setObjectName("normOffRadio")
        self.normOffRadio.setText(translate("Form", "Off"))
        self.gridLayout_2.addWidget(self.normOffRadio, 0, 3, 1, 1)

        self.label_3 = QtWidgets.QLabel(self.normGroup)
        self.label_3.setFont(boldFont)
        self.label_3.setObjectName("label_3")
        self.label_3.setText(translate("Form", "Mean:"))
        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)

        self.normROICheck = QtWidgets.QCheckBox(self.normGroup)
        self.normROICheck.setObjectName("normROICheck")
        self.normROICheck.setText(translate("Form", "ROI"))
        self.gridLayout_2.addWidget(self.normROICheck, 1, 1, 1, 1)

        self.normFrameCheck = QtWidgets.QCheckBox(self.normGroup)
        self.normFrameCheck.setObjectName("normFrameCheck")
        self.normFrameCheck.setText(translate("Form", "Frame"))
        self.gridLayout_2.addWidget(self.normFrameCheck, 1, 2, 1, 1)

        self.normTimeRangeCheck = QtWidgets.QCheckBox(self.normGroup)
        self.normTimeRangeCheck.setObjectName("normTimeRangeCheck")
        self.normTimeRangeCheck.setText(translate("Form", "Time range"))
        self.gridLayout_2.addWidget(self.normTimeRangeCheck, 1, 3, 1, 1)

        self.label_4 = QtWidgets.QLabel(self.normGroup)
        self.label_4.setFont(boldFont)
        self.label_4.setObjectName("label_4")
        self.label_4.setText(translate("Form", "Blur:"))
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)

        self.label_8 = QtWidgets.QLabel(self.normGroup)
        self.label_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.label_8.setText(translate("Form", "X"))
        self.gridLayout_2.addWidget(self.label_8, 2, 1, 1, 1)

        self.normXBlurSpin = QtWidgets.QDoubleSpinBox(self.normGroup)
        self.normXBlurSpin.setObjectName("normXBlurSpin")
        self.gridLayout_2.addWidget(self.normXBlurSpin, 2, 2, 1, 1)

        self.label_9 = QtWidgets.QLabel(self.normGroup)
        self.label_9.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.label_9.setText(translate("Form", "Y"))
        self.gridLayout_2.addWidget(self.label_9, 2, 3, 1, 1)

        self.normYBlurSpin = QtWidgets.QDoubleSpinBox(self.normGroup)
        self.normYBlurSpin.setObjectName("normYBlurSpin")
        self.gridLayout_2.addWidget(self.normYBlurSpin, 2, 4, 1, 1)

        self.label_10 = QtWidgets.QLabel(self.normGroup)
        self.label_10.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.label_10.setText(translate("Form", "T"))
        self.gridLayout_2.addWidget(self.label_10, 2, 5, 1, 1)

        self.normTBlurSpin = QtWidgets.QDoubleSpinBox(self.normGroup)
        self.normTBlurSpin.setObjectName("normTBlurSpin")
        self.gridLayout_2.addWidget(self.normTBlurSpin, 2, 6, 1, 1)

        self.gridLayout_3.addWidget(self.normGroup, 1, 0, 1, 1)

        QtCore.QMetaObject.connectSlotsByName(Form)
