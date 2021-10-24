# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'axisCtrlTemplate.ui'
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
        Form.resize(186, 154)
        Form.setMaximumSize(QSize(200, 16777215))
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 7, 0, 1, 2)

        self.linkCombo = QComboBox(Form)
        self.linkCombo.setObjectName(u"linkCombo")
        self.linkCombo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.gridLayout.addWidget(self.linkCombo, 7, 2, 1, 2)

        self.autoPercentSpin = QSpinBox(Form)
        self.autoPercentSpin.setObjectName(u"autoPercentSpin")
        self.autoPercentSpin.setEnabled(True)
        self.autoPercentSpin.setMinimum(1)
        self.autoPercentSpin.setMaximum(100)
        self.autoPercentSpin.setSingleStep(1)
        self.autoPercentSpin.setValue(100)

        self.gridLayout.addWidget(self.autoPercentSpin, 2, 2, 1, 2)

        self.autoRadio = QRadioButton(Form)
        self.autoRadio.setObjectName(u"autoRadio")
        self.autoRadio.setChecked(True)

        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)

        self.manualRadio = QRadioButton(Form)
        self.manualRadio.setObjectName(u"manualRadio")

        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)

        self.minText = QLineEdit(Form)
        self.minText.setObjectName(u"minText")

        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)

        self.maxText = QLineEdit(Form)
        self.maxText.setObjectName(u"maxText")

        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)

        self.invertCheck = QCheckBox(Form)
        self.invertCheck.setObjectName(u"invertCheck")

        self.gridLayout.addWidget(self.invertCheck, 5, 0, 1, 4)

        self.mouseCheck = QCheckBox(Form)
        self.mouseCheck.setObjectName(u"mouseCheck")
        self.mouseCheck.setChecked(True)

        self.gridLayout.addWidget(self.mouseCheck, 6, 0, 1, 4)

        self.visibleOnlyCheck = QCheckBox(Form)
        self.visibleOnlyCheck.setObjectName(u"visibleOnlyCheck")

        self.gridLayout.addWidget(self.visibleOnlyCheck, 3, 2, 1, 2)

        self.autoPanCheck = QCheckBox(Form)
        self.autoPanCheck.setObjectName(u"autoPanCheck")

        self.gridLayout.addWidget(self.autoPanCheck, 4, 2, 1, 2)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PyQtGraph", None))
        self.label.setText(QCoreApplication.translate("Form", u"Link Axis:", None))
#if QT_CONFIG(tooltip)
        self.linkCombo.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Links this axis with another view. When linked, both views will display the same data range.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.autoPercentSpin.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Percent of data to be visible when auto-scaling. It may be useful to decrease this value for data with spiky noise.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.autoPercentSpin.setSuffix(QCoreApplication.translate("Form", u"%", None))
#if QT_CONFIG(tooltip)
        self.autoRadio.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.autoRadio.setText(QCoreApplication.translate("Form", u"Auto", None))
#if QT_CONFIG(tooltip)
        self.manualRadio.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.manualRadio.setText(QCoreApplication.translate("Form", u"Manual", None))
#if QT_CONFIG(tooltip)
        self.minText.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.minText.setText(QCoreApplication.translate("Form", u"0", None))
#if QT_CONFIG(tooltip)
        self.maxText.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.maxText.setText(QCoreApplication.translate("Form", u"0", None))
#if QT_CONFIG(tooltip)
        self.invertCheck.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Inverts the display of this axis. (+y points downward instead of upward)</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.invertCheck.setText(QCoreApplication.translate("Form", u"Invert Axis", None))
#if QT_CONFIG(tooltip)
        self.mouseCheck.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Enables mouse interaction (panning, scaling) for this axis.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.mouseCheck.setText(QCoreApplication.translate("Form", u"Mouse Enabled", None))
#if QT_CONFIG(tooltip)
        self.visibleOnlyCheck.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>When checked, the axis will only auto-scale to data that is visible along the orthogonal axis.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.visibleOnlyCheck.setText(QCoreApplication.translate("Form", u"Visible Data Only", None))
#if QT_CONFIG(tooltip)
        self.autoPanCheck.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>When checked, the axis will automatically pan to center on the current data, but the scale along this axis will not change.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.autoPanCheck.setText(QCoreApplication.translate("Form", u"Auto Pan Only", None))
    # retranslateUi

