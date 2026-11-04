from ..Qt import QtCore, QtWidgets
from ..parametertree import ParameterTree

translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(241, 367)
        Form.setWindowTitle(translate("Form", "Export"))

        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.label.setText(translate("Form", "Item to export:"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)

        self.itemTree = QtWidgets.QTreeWidget(Form)
        self.itemTree.setObjectName("itemTree")
        self.itemTree.headerItem().setText(0, "1")
        self.itemTree.header().setVisible(False)
        self.gridLayout.addWidget(self.itemTree, 1, 0, 1, 3)

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.label_2.setText(translate("Form", "Export format"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 3)

        self.formatList = QtWidgets.QListWidget(Form)
        self.formatList.setObjectName("formatList")
        self.gridLayout.addWidget(self.formatList, 3, 0, 1, 3)

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.label_3.setText(translate("Form", "Export options"))
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 3)

        self.paramTree = ParameterTree(Form)
        self.paramTree.setColumnCount(2)
        self.paramTree.setObjectName("paramTree")
        self.paramTree.headerItem().setText(0, "1")
        self.paramTree.header().setVisible(False)
        self.gridLayout.addWidget(self.paramTree, 5, 0, 1, 3)

        self.copyBtn = QtWidgets.QPushButton(Form)
        self.copyBtn.setObjectName("copyBtn")
        self.copyBtn.setText(translate("Form", "Copy"))
        self.gridLayout.addWidget(self.copyBtn, 6, 0, 1, 1)

        self.exportBtn = QtWidgets.QPushButton(Form)
        self.exportBtn.setObjectName("exportBtn")
        self.exportBtn.setText(translate("Form", "Export"))
        self.gridLayout.addWidget(self.exportBtn, 6, 1, 1, 1)

        self.closeBtn = QtWidgets.QPushButton(Form)
        self.closeBtn.setObjectName("closeBtn")
        self.closeBtn.setText(translate("Form", "Close"))
        self.gridLayout.addWidget(self.closeBtn, 6, 2, 1, 1)

        QtCore.QMetaObject.connectSlotsByName(Form)
