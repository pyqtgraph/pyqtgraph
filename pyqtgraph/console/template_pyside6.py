# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'template.ui'
##
## Created by: Qt User Interface Compiler version 6.0.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .CmdInput import CmdInput


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(739, 497)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.output = QPlainTextEdit(self.layoutWidget)
        self.output.setObjectName(u"output")
        font = QFont()
        font.setFamily(u"Monospace")
        self.output.setFont(font)
        self.output.setReadOnly(True)

        self.verticalLayout.addWidget(self.output)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input = CmdInput(self.layoutWidget)
        self.input.setObjectName(u"input")

        self.horizontalLayout.addWidget(self.input)

        self.historyBtn = QPushButton(self.layoutWidget)
        self.historyBtn.setObjectName(u"historyBtn")
        self.historyBtn.setCheckable(True)

        self.horizontalLayout.addWidget(self.historyBtn)

        self.exceptionBtn = QPushButton(self.layoutWidget)
        self.exceptionBtn.setObjectName(u"exceptionBtn")
        self.exceptionBtn.setCheckable(True)

        self.horizontalLayout.addWidget(self.exceptionBtn)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.splitter.addWidget(self.layoutWidget)
        self.historyList = QListWidget(self.splitter)
        self.historyList.setObjectName(u"historyList")
        self.historyList.setFont(font)
        self.splitter.addWidget(self.historyList)
        self.exceptionGroup = QGroupBox(self.splitter)
        self.exceptionGroup.setObjectName(u"exceptionGroup")
        self.gridLayout_2 = QGridLayout(self.exceptionGroup)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(2)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setContentsMargins(-1, 0, -1, 0)
        self.clearExceptionBtn = QPushButton(self.exceptionGroup)
        self.clearExceptionBtn.setObjectName(u"clearExceptionBtn")
        self.clearExceptionBtn.setEnabled(False)

        self.gridLayout_2.addWidget(self.clearExceptionBtn, 0, 6, 1, 1)

        self.catchAllExceptionsBtn = QPushButton(self.exceptionGroup)
        self.catchAllExceptionsBtn.setObjectName(u"catchAllExceptionsBtn")
        self.catchAllExceptionsBtn.setCheckable(True)

        self.gridLayout_2.addWidget(self.catchAllExceptionsBtn, 0, 1, 1, 1)

        self.catchNextExceptionBtn = QPushButton(self.exceptionGroup)
        self.catchNextExceptionBtn.setObjectName(u"catchNextExceptionBtn")
        self.catchNextExceptionBtn.setCheckable(True)

        self.gridLayout_2.addWidget(self.catchNextExceptionBtn, 0, 0, 1, 1)

        self.onlyUncaughtCheck = QCheckBox(self.exceptionGroup)
        self.onlyUncaughtCheck.setObjectName(u"onlyUncaughtCheck")
        self.onlyUncaughtCheck.setChecked(True)

        self.gridLayout_2.addWidget(self.onlyUncaughtCheck, 0, 4, 1, 1)

        self.exceptionStackList = QListWidget(self.exceptionGroup)
        self.exceptionStackList.setObjectName(u"exceptionStackList")
        self.exceptionStackList.setAlternatingRowColors(True)

        self.gridLayout_2.addWidget(self.exceptionStackList, 2, 0, 1, 7)

        self.runSelectedFrameCheck = QCheckBox(self.exceptionGroup)
        self.runSelectedFrameCheck.setObjectName(u"runSelectedFrameCheck")
        self.runSelectedFrameCheck.setChecked(True)

        self.gridLayout_2.addWidget(self.runSelectedFrameCheck, 3, 0, 1, 7)

        self.exceptionInfoLabel = QLabel(self.exceptionGroup)
        self.exceptionInfoLabel.setObjectName(u"exceptionInfoLabel")
        self.exceptionInfoLabel.setWordWrap(True)

        self.gridLayout_2.addWidget(self.exceptionInfoLabel, 1, 0, 1, 7)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 5, 1, 1)

        self.label = QLabel(self.exceptionGroup)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 2, 1, 1)

        self.filterText = QLineEdit(self.exceptionGroup)
        self.filterText.setObjectName(u"filterText")

        self.gridLayout_2.addWidget(self.filterText, 0, 3, 1, 1)

        self.splitter.addWidget(self.exceptionGroup)

        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Console", None))
        self.historyBtn.setText(QCoreApplication.translate("Form", u"History..", None))
        self.exceptionBtn.setText(QCoreApplication.translate("Form", u"Exceptions..", None))
        self.exceptionGroup.setTitle(QCoreApplication.translate("Form", u"Exception Handling", None))
        self.clearExceptionBtn.setText(QCoreApplication.translate("Form", u"Clear Stack", None))
        self.catchAllExceptionsBtn.setText(QCoreApplication.translate("Form", u"Show All Exceptions", None))
        self.catchNextExceptionBtn.setText(QCoreApplication.translate("Form", u"Show Next Exception", None))
        self.onlyUncaughtCheck.setText(QCoreApplication.translate("Form", u"Only Uncaught Exceptions", None))
        self.runSelectedFrameCheck.setText(QCoreApplication.translate("Form", u"Run commands in selected stack frame", None))
        self.exceptionInfoLabel.setText(QCoreApplication.translate("Form", u"Stack Trace", None))
        self.label.setText(QCoreApplication.translate("Form", u"Filter (regex):", None))
    # retranslateUi

