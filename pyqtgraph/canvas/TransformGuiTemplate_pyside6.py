
################################################################################
## Form generated from reading UI file 'TransformGuiTemplate.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
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
        Form.resize(224, 117)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.translateLabel = QLabel(Form)
        self.translateLabel.setObjectName(u"translateLabel")

        self.verticalLayout.addWidget(self.translateLabel)

        self.rotateLabel = QLabel(Form)
        self.rotateLabel.setObjectName(u"rotateLabel")

        self.verticalLayout.addWidget(self.rotateLabel)

        self.scaleLabel = QLabel(Form)
        self.scaleLabel.setObjectName(u"scaleLabel")

        self.verticalLayout.addWidget(self.scaleLabel)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.mirrorImageBtn = QPushButton(Form)
        self.mirrorImageBtn.setObjectName(u"mirrorImageBtn")

        self.horizontalLayout.addWidget(self.mirrorImageBtn)

        self.reflectImageBtn = QPushButton(Form)
        self.reflectImageBtn.setObjectName(u"reflectImageBtn")

        self.horizontalLayout.addWidget(self.reflectImageBtn)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.translateLabel.setText(QCoreApplication.translate("Form", u"Translate:", None))
        self.rotateLabel.setText(QCoreApplication.translate("Form", u"Rotate:", None))
        self.scaleLabel.setText(QCoreApplication.translate("Form", u"Scale:", None))
#if QT_CONFIG(tooltip)
        self.mirrorImageBtn.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.mirrorImageBtn.setText(QCoreApplication.translate("Form", u"Mirror", None))
        self.reflectImageBtn.setText(QCoreApplication.translate("Form", u"Reflect", None))
    # retranslateUi
