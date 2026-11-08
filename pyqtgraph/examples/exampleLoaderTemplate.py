from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(846, 552)
        Form.setWindowTitle(translate("Form", "PyQtGraph"))

        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setObjectName("splitter")

        # -- left pane: filter, example tree, load button --
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.exampleFilter = QtWidgets.QLineEdit(self.layoutWidget)
        self.exampleFilter.setObjectName("exampleFilter")
        self.exampleFilter.setPlaceholderText(translate("Form", "Type to filter..."))
        self.gridLayout.addWidget(self.exampleFilter, 0, 0, 1, 2)

        self.searchFiles = QtWidgets.QComboBox(self.layoutWidget)
        self.searchFiles.setObjectName("searchFiles")
        self.searchFiles.addItem(translate("Form", "Title Search"))
        self.searchFiles.addItem(translate("Form", "Content Search"))
        self.gridLayout.addWidget(self.searchFiles, 1, 0, 1, 2)

        self.exampleTree = QtWidgets.QTreeWidget(self.layoutWidget)
        self.exampleTree.setObjectName("exampleTree")
        self.exampleTree.headerItem().setText(0, "1")
        self.exampleTree.header().setVisible(False)
        self.gridLayout.addWidget(self.exampleTree, 3, 0, 1, 2)

        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.label.setText(translate("Form", "Qt Library:"))
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)

        self.qtLibCombo = QtWidgets.QComboBox(self.layoutWidget)
        self.qtLibCombo.setObjectName("qtLibCombo")
        self.gridLayout.addWidget(self.qtLibCombo, 4, 1, 1, 1)

        self.loadBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.loadBtn.setObjectName("loadBtn")
        self.loadBtn.setText(translate("Form", "Run Example"))
        self.gridLayout.addWidget(self.loadBtn, 6, 1, 1, 1)

        # -- right pane: loaded file name + source preview --
        self.layoutWidget1 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.loadedFileLabel = QtWidgets.QLabel(self.layoutWidget1)
        boldFont = QtGui.QFont()
        boldFont.setBold(True)
        self.loadedFileLabel.setFont(boldFont)
        self.loadedFileLabel.setText("")
        self.loadedFileLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.loadedFileLabel.setObjectName("loadedFileLabel")
        self.verticalLayout.addWidget(self.loadedFileLabel)

        self.codeView = QtWidgets.QPlainTextEdit(self.layoutWidget1)
        codeFont = QtGui.QFont()
        codeFont.setFamily("Courier New")
        self.codeView.setFont(codeFont)
        self.codeView.setObjectName("codeView")
        self.verticalLayout.addWidget(self.codeView)

        self.gridLayout_2.addWidget(self.splitter, 1, 0, 1, 1)

        QtCore.QMetaObject.connectSlotsByName(Form)
