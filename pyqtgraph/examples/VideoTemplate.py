from pyqtgraph import GradientWidget, GraphicsView, SpinBox
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.widgets.RawImageWidget import RawImageWidget

translate = QtCore.QCoreApplication.translate


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(695, 798)
        MainWindow.setWindowTitle(translate("MainWindow", "MainWindow"))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.fpsLabel = QtWidgets.QLabel(self.centralwidget)
        fpsFont = QtGui.QFont()
        fpsFont.setPointSize(12)
        self.fpsLabel.setFont(fpsFont)
        self.fpsLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.fpsLabel.setObjectName("fpsLabel")
        self.fpsLabel.setText(translate("MainWindow", "FPS"))
        self.gridLayout_2.addWidget(self.fpsLabel, 0, 0, 1, 4)

        # -- render-mode radio buttons + view stack --
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.gfxRadio = QtWidgets.QRadioButton(self.centralwidget)
        self.gfxRadio.setChecked(True)
        self.gfxRadio.setObjectName("gfxRadio")
        self.gfxRadio.setText(translate("MainWindow", "GraphicsView + ImageItem"))
        self.gridLayout.addWidget(self.gfxRadio, 2, 0, 1, 1)

        self.rawRadio = QtWidgets.QRadioButton(self.centralwidget)
        self.rawRadio.setObjectName("rawRadio")
        self.rawRadio.setText(translate("MainWindow", "RawImageWidget"))
        self.gridLayout.addWidget(self.rawRadio, 3, 0, 1, 1)

        self.rawGLRadio = QtWidgets.QRadioButton(self.centralwidget)
        self.rawGLRadio.setObjectName("rawGLRadio")
        self.rawGLRadio.setText(translate("MainWindow", "RawGLImageWidget"))
        self.gridLayout.addWidget(self.rawGLRadio, 4, 0, 1, 1)

        self.stack = QtWidgets.QStackedWidget(self.centralwidget)
        self.stack.setObjectName("stack")

        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.graphicsView = GraphicsView(self.page)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_3.addWidget(self.graphicsView, 0, 0, 1, 1)
        self.stack.addWidget(self.page)

        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.rawImg = RawImageWidget(self.page_2)
        rawImgSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        rawImgSizePolicy.setHorizontalStretch(0)
        rawImgSizePolicy.setVerticalStretch(0)
        rawImgSizePolicy.setHeightForWidth(self.rawImg.sizePolicy().hasHeightForWidth())
        self.rawImg.setSizePolicy(rawImgSizePolicy)
        self.rawImg.setObjectName("rawImg")
        self.gridLayout_4.addWidget(self.rawImg, 0, 0, 1, 1)
        self.stack.addWidget(self.page_2)

        self.gridLayout.addWidget(self.stack, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 4)

        # -- image size / dtype --
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.framesSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.framesSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.framesSpin.setProperty("value", 10)
        self.framesSpin.setObjectName("framesSpin")
        self.horizontalLayout_4.addWidget(self.framesSpin)

        self.widthSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.widthSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.widthSpin.setMaximum(10000)
        self.widthSpin.setProperty("value", 512)
        self.widthSpin.setObjectName("widthSpin")
        self.horizontalLayout_4.addWidget(self.widthSpin)

        self.heightSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.heightSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.heightSpin.setMaximum(10000)
        self.heightSpin.setProperty("value", 512)
        self.heightSpin.setObjectName("heightSpin")
        self.horizontalLayout_4.addWidget(self.heightSpin)

        self.gridLayout_2.addLayout(self.horizontalLayout_4, 2, 1, 1, 2)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.label_5.setText(translate("MainWindow", "Image size"))
        self.gridLayout_2.addWidget(self.label_5, 2, 0, 1, 1)

        self.sizeLabel = QtWidgets.QLabel(self.centralwidget)
        self.sizeLabel.setText("")
        self.sizeLabel.setObjectName("sizeLabel")
        self.gridLayout_2.addWidget(self.sizeLabel, 2, 3, 1, 1)

        sizeSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(sizeSpacer, 3, 3, 1, 1)

        self.rgbCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.rgbCheck.setObjectName("rgbCheck")
        self.rgbCheck.setText(translate("MainWindow", "RGB"))
        self.gridLayout_2.addWidget(self.rgbCheck, 3, 1, 1, 1)

        self.dtypeCombo = QtWidgets.QComboBox(self.centralwidget)
        self.dtypeCombo.setObjectName("dtypeCombo")
        self.dtypeCombo.addItem(translate("MainWindow", "uint8"))
        self.dtypeCombo.addItem(translate("MainWindow", "uint16"))
        self.dtypeCombo.addItem(translate("MainWindow", "float"))
        self.gridLayout_2.addWidget(self.dtypeCombo, 3, 2, 1, 1)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setText(translate("MainWindow", "Data type"))
        self.gridLayout_2.addWidget(self.label, 3, 0, 1, 1)

        # -- scaling / levels --
        self.scaleCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.scaleCheck.setObjectName("scaleCheck")
        self.scaleCheck.setText(translate("MainWindow", "Scale Data"))
        self.gridLayout_2.addWidget(self.scaleCheck, 4, 0, 1, 1)

        self.rgbLevelsCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.rgbLevelsCheck.setObjectName("rgbLevelsCheck")
        self.rgbLevelsCheck.setText(translate("MainWindow", "RGB"))
        self.gridLayout_2.addWidget(self.rgbLevelsCheck, 4, 1, 1, 1)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.minSpin1 = SpinBox(self.centralwidget)
        self.minSpin1.setObjectName("minSpin1")
        self.horizontalLayout.addWidget(self.minSpin1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_2.setText(translate("MainWindow", "<--->"))
        self.horizontalLayout.addWidget(self.label_2)
        self.maxSpin1 = SpinBox(self.centralwidget)
        self.maxSpin1.setObjectName("maxSpin1")
        self.horizontalLayout.addWidget(self.maxSpin1)
        self.gridLayout_2.addLayout(self.horizontalLayout, 4, 2, 1, 1)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.minSpin2 = SpinBox(self.centralwidget)
        self.minSpin2.setEnabled(False)
        self.minSpin2.setObjectName("minSpin2")
        self.horizontalLayout_2.addWidget(self.minSpin2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_3.setText(translate("MainWindow", "<--->"))
        self.horizontalLayout_2.addWidget(self.label_3)
        self.maxSpin2 = SpinBox(self.centralwidget)
        self.maxSpin2.setEnabled(False)
        self.maxSpin2.setObjectName("maxSpin2")
        self.horizontalLayout_2.addWidget(self.maxSpin2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 5, 2, 1, 1)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.minSpin3 = SpinBox(self.centralwidget)
        self.minSpin3.setEnabled(False)
        self.minSpin3.setObjectName("minSpin3")
        self.horizontalLayout_3.addWidget(self.minSpin3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_4.setText(translate("MainWindow", "<--->"))
        self.horizontalLayout_3.addWidget(self.label_4)
        self.maxSpin3 = SpinBox(self.centralwidget)
        self.maxSpin3.setEnabled(False)
        self.maxSpin3.setObjectName("maxSpin3")
        self.horizontalLayout_3.addWidget(self.maxSpin3)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 6, 2, 1, 1)

        # -- LUT / gradient --
        self.lutCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.lutCheck.setObjectName("lutCheck")
        self.lutCheck.setText(translate("MainWindow", "Use Lookup  Table"))
        self.gridLayout_2.addWidget(self.lutCheck, 7, 0, 1, 1)

        self.alphaCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.alphaCheck.setObjectName("alphaCheck")
        self.alphaCheck.setText(translate("MainWindow", "alpha"))
        self.gridLayout_2.addWidget(self.alphaCheck, 7, 1, 1, 1)

        self.gradient = GradientWidget(self.centralwidget)
        gradientSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        gradientSizePolicy.setHorizontalStretch(0)
        gradientSizePolicy.setVerticalStretch(0)
        gradientSizePolicy.setHeightForWidth(self.gradient.sizePolicy().hasHeightForWidth())
        self.gradient.setSizePolicy(gradientSizePolicy)
        self.gradient.setObjectName("gradient")
        self.gridLayout_2.addWidget(self.gradient, 7, 2, 1, 2)

        # -- backend toggles --
        self.downsampleCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.downsampleCheck.setObjectName("downsampleCheck")
        self.downsampleCheck.setText(translate("MainWindow", "Auto downsample"))
        self.gridLayout_2.addWidget(self.downsampleCheck, 8, 0, 1, 2)

        self.cudaCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.cudaCheck.setObjectName("cudaCheck")
        self.cudaCheck.setText(translate("MainWindow", "Use CUDA (GPU) if available"))
        self.gridLayout_2.addWidget(self.cudaCheck, 9, 0, 1, 2)

        self.numbaCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.numbaCheck.setObjectName("numbaCheck")
        self.numbaCheck.setText(translate("MainWindow", "Use Numba if available"))
        self.gridLayout_2.addWidget(self.numbaCheck, 9, 2, 1, 2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.stack.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
