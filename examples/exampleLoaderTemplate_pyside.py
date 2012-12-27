# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './examples/exampleLoaderTemplate.ui'
#
# Created: Mon Dec 24 00:33:39 2012
#      by: pyside-uic 0.2.13 running on PySide 1.1.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(762, 302)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.exampleTree = QtGui.QTreeWidget(self.layoutWidget)
        self.exampleTree.setObjectName("exampleTree")
        self.exampleTree.headerItem().setText(0, "1")
        self.exampleTree.header().setVisible(False)
        self.verticalLayout.addWidget(self.exampleTree)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pyqtCheck = QtGui.QCheckBox(self.layoutWidget)
        self.pyqtCheck.setObjectName("pyqtCheck")
        self.horizontalLayout.addWidget(self.pyqtCheck)
        self.pysideCheck = QtGui.QCheckBox(self.layoutWidget)
        self.pysideCheck.setObjectName("pysideCheck")
        self.horizontalLayout.addWidget(self.pysideCheck)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.forceGraphicsCheck = QtGui.QCheckBox(self.layoutWidget)
        self.forceGraphicsCheck.setObjectName("forceGraphicsCheck")
        self.horizontalLayout_2.addWidget(self.forceGraphicsCheck)
        self.forceGraphicsCombo = QtGui.QComboBox(self.layoutWidget)
        self.forceGraphicsCombo.setObjectName("forceGraphicsCombo")
        self.forceGraphicsCombo.addItem("")
        self.forceGraphicsCombo.addItem("")
        self.forceGraphicsCombo.addItem("")
        self.horizontalLayout_2.addWidget(self.forceGraphicsCombo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.loadBtn = QtGui.QPushButton(self.layoutWidget)
        self.loadBtn.setObjectName("loadBtn")
        self.verticalLayout.addWidget(self.loadBtn)
        self.codeView = QtGui.QTextBrowser(self.splitter)
        font = QtGui.QFont()
        font.setFamily("Monospace")
        font.setPointSize(10)
        self.codeView.setFont(font)
        self.codeView.setObjectName("codeView")
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.pyqtCheck.setText(QtGui.QApplication.translate("Form", "Force PyQt", None, QtGui.QApplication.UnicodeUTF8))
        self.pysideCheck.setText(QtGui.QApplication.translate("Form", "Force PySide", None, QtGui.QApplication.UnicodeUTF8))
        self.forceGraphicsCheck.setText(QtGui.QApplication.translate("Form", "Force Graphics System:", None, QtGui.QApplication.UnicodeUTF8))
        self.forceGraphicsCombo.setItemText(0, QtGui.QApplication.translate("Form", "native", None, QtGui.QApplication.UnicodeUTF8))
        self.forceGraphicsCombo.setItemText(1, QtGui.QApplication.translate("Form", "raster", None, QtGui.QApplication.UnicodeUTF8))
        self.forceGraphicsCombo.setItemText(2, QtGui.QApplication.translate("Form", "opengl", None, QtGui.QApplication.UnicodeUTF8))
        self.loadBtn.setText(QtGui.QApplication.translate("Form", "Load Example", None, QtGui.QApplication.UnicodeUTF8))

