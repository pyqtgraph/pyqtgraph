# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TransformGuiTemplate.ui'
#
# Created: Sun Sep 18 19:18:41 2016
#      by: pyside2-uic  running on PySide2 2.0.0~alpha0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(224, 117)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.translateLabel = QtWidgets.QLabel(Form)
        self.translateLabel.setObjectName("translateLabel")
        self.verticalLayout.addWidget(self.translateLabel)
        self.rotateLabel = QtWidgets.QLabel(Form)
        self.rotateLabel.setObjectName("rotateLabel")
        self.verticalLayout.addWidget(self.rotateLabel)
        self.scaleLabel = QtWidgets.QLabel(Form)
        self.scaleLabel.setObjectName("scaleLabel")
        self.verticalLayout.addWidget(self.scaleLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mirrorImageBtn = QtWidgets.QPushButton(Form)
        self.mirrorImageBtn.setToolTip("")
        self.mirrorImageBtn.setObjectName("mirrorImageBtn")
        self.horizontalLayout.addWidget(self.mirrorImageBtn)
        self.reflectImageBtn = QtWidgets.QPushButton(Form)
        self.reflectImageBtn.setObjectName("reflectImageBtn")
        self.horizontalLayout.addWidget(self.reflectImageBtn)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.translateLabel.setText(QtWidgets.QApplication.translate("Form", "Translate:", None, -1))
        self.rotateLabel.setText(QtWidgets.QApplication.translate("Form", "Rotate:", None, -1))
        self.scaleLabel.setText(QtWidgets.QApplication.translate("Form", "Scale:", None, -1))
        self.mirrorImageBtn.setText(QtWidgets.QApplication.translate("Form", "Mirror", None, -1))
        self.reflectImageBtn.setText(QtWidgets.QApplication.translate("Form", "Reflect", None, -1))

