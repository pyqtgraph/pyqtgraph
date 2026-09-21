from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.FeedbackButton import FeedbackButton
from ..widgets.TreeWidget import TreeWidget

translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(217, 499)
        Form.setWindowTitle(translate("Form", "PyQtGraph"))

        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.fileNameLabel = QtWidgets.QLabel(Form)
        boldFont = QtGui.QFont()
        boldFont.setBold(True)
        self.fileNameLabel.setFont(boldFont)
        self.fileNameLabel.setText("")
        self.fileNameLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.fileNameLabel.setObjectName("fileNameLabel")
        self.gridLayout.addWidget(self.fileNameLabel, 0, 1, 1, 1)

        self.loadBtn = QtWidgets.QPushButton(Form)
        self.loadBtn.setObjectName("loadBtn")
        self.loadBtn.setText(translate("Form", "Load.."))
        self.gridLayout.addWidget(self.loadBtn, 1, 0, 1, 1)

        self.saveBtn = FeedbackButton(Form)
        self.saveBtn.setObjectName("saveBtn")
        self.saveBtn.setText(translate("Form", "Save"))
        self.gridLayout.addWidget(self.saveBtn, 1, 1, 1, 2)

        self.saveAsBtn = FeedbackButton(Form)
        self.saveAsBtn.setObjectName("saveAsBtn")
        self.saveAsBtn.setText(translate("Form", "As.."))
        self.gridLayout.addWidget(self.saveAsBtn, 1, 3, 1, 1)

        self.ctrlList = TreeWidget(Form)
        self.ctrlList.setObjectName("ctrlList")
        self.ctrlList.headerItem().setText(0, "1")
        self.ctrlList.header().setVisible(False)
        self.ctrlList.header().setStretchLastSection(False)
        self.gridLayout.addWidget(self.ctrlList, 3, 0, 1, 4)

        self.reloadBtn = FeedbackButton(Form)
        self.reloadBtn.setCheckable(False)
        self.reloadBtn.setFlat(False)
        self.reloadBtn.setObjectName("reloadBtn")
        self.reloadBtn.setText(translate("Form", "Reload Libs"))
        self.gridLayout.addWidget(self.reloadBtn, 4, 0, 1, 2)

        self.showChartBtn = QtWidgets.QPushButton(Form)
        self.showChartBtn.setCheckable(True)
        self.showChartBtn.setObjectName("showChartBtn")
        self.showChartBtn.setText(translate("Form", "Flowchart"))
        self.gridLayout.addWidget(self.showChartBtn, 4, 2, 1, 2)

        QtCore.QMetaObject.connectSlotsByName(Form)
