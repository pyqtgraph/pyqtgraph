# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './examples/exampleLoaderTemplate.ui'
#
# Created: Mon Dec 24 00:33:38 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(762, 302)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.exampleTree = QtGui.QTreeWidget(self.layoutWidget)
        self.exampleTree.setObjectName(_fromUtf8("exampleTree"))
        self.exampleTree.headerItem().setText(0, _fromUtf8("1"))
        self.exampleTree.header().setVisible(False)
        self.verticalLayout.addWidget(self.exampleTree)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pyqtCheck = QtGui.QCheckBox(self.layoutWidget)
        self.pyqtCheck.setObjectName(_fromUtf8("pyqtCheck"))
        self.horizontalLayout.addWidget(self.pyqtCheck)
        self.pysideCheck = QtGui.QCheckBox(self.layoutWidget)
        self.pysideCheck.setObjectName(_fromUtf8("pysideCheck"))
        self.horizontalLayout.addWidget(self.pysideCheck)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.forceGraphicsCheck = QtGui.QCheckBox(self.layoutWidget)
        self.forceGraphicsCheck.setObjectName(_fromUtf8("forceGraphicsCheck"))
        self.horizontalLayout_2.addWidget(self.forceGraphicsCheck)
        self.forceGraphicsCombo = QtGui.QComboBox(self.layoutWidget)
        self.forceGraphicsCombo.setObjectName(_fromUtf8("forceGraphicsCombo"))
        self.forceGraphicsCombo.addItem(_fromUtf8(""))
        self.forceGraphicsCombo.addItem(_fromUtf8(""))
        self.forceGraphicsCombo.addItem(_fromUtf8(""))
        self.horizontalLayout_2.addWidget(self.forceGraphicsCombo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.loadBtn = QtGui.QPushButton(self.layoutWidget)
        self.loadBtn.setObjectName(_fromUtf8("loadBtn"))
        self.verticalLayout.addWidget(self.loadBtn)
        self.codeView = QtGui.QTextBrowser(self.splitter)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        font.setPointSize(10)
        self.codeView.setFont(font)
        self.codeView.setObjectName(_fromUtf8("codeView"))
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

