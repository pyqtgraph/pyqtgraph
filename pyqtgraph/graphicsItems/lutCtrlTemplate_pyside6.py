# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'lutCtrlTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.0.0
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
        Form.resize(200, 62)
        Form.setMaximumSize(QSize(200, 16777215))
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.manualRadio = QRadioButton(Form)
        self.manualRadio.setObjectName(u"manualRadio")

        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)

        self.logCheck = QCheckBox(Form)
        self.logCheck.setObjectName(u"logCheck")

        self.gridLayout.addWidget(self.logCheck, 3, 0, 1, 4)

        self.autoRadio = QRadioButton(Form)
        self.autoRadio.setObjectName(u"autoRadio")
        self.autoRadio.setChecked(True)

        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)

        self.minText = QLineEdit(Form)
        self.minText.setObjectName(u"minText")

        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)

        self.maxText = QLineEdit(Form)
        self.maxText.setObjectName(u"maxText")

        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
#if QT_CONFIG(tooltip)
        self.manualRadio.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.manualRadio.setText(QCoreApplication.translate("Form", u"Manual", None))
#if QT_CONFIG(tooltip)
        self.logCheck.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Use log scale</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.logCheck.setText(QCoreApplication.translate("Form", u"Log scale", None))
#if QT_CONFIG(tooltip)
        self.autoRadio.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.autoRadio.setText(QCoreApplication.translate("Form", u"Auto", None))
#if QT_CONFIG(tooltip)
        self.minText.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.minText.setText(QCoreApplication.translate("Form", u"0", None))
#if QT_CONFIG(tooltip)
        self.maxText.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.maxText.setText(QCoreApplication.translate("Form", u"0", None))
    # retranslateUi

