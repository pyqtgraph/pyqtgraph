# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/graphicsItems/ViewBox/axisCtrlTemplate.ui',
# licensing of 'C:/Repos/TestLibs/pyqtgraph/tools\../pyqtgraph/graphicsItems/ViewBox/axisCtrlTemplate.ui' applies.
#
# Created: Wed Jan 30 12:16:52 2019
#      by: pyside2-uic  running on PySide2 5.12.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(186, 154)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 7, 0, 1, 2)
        self.linkCombo = QtWidgets.QComboBox(Form)
        self.linkCombo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.linkCombo.setObjectName("linkCombo")
        self.gridLayout.addWidget(self.linkCombo, 7, 2, 1, 2)
        self.autoPercentSpin = QtWidgets.QSpinBox(Form)
        self.autoPercentSpin.setEnabled(True)
        self.autoPercentSpin.setMinimum(1)
        self.autoPercentSpin.setMaximum(100)
        self.autoPercentSpin.setSingleStep(1)
        self.autoPercentSpin.setProperty("value", 100)
        self.autoPercentSpin.setObjectName("autoPercentSpin")
        self.gridLayout.addWidget(self.autoPercentSpin, 2, 2, 1, 2)
        self.autoRadio = QtWidgets.QRadioButton(Form)
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName("autoRadio")
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)
        self.manualRadio = QtWidgets.QRadioButton(Form)
        self.manualRadio.setObjectName("manualRadio")
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)
        self.minText = QtWidgets.QLineEdit(Form)
        self.minText.setObjectName("minText")
        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)
        self.maxText = QtWidgets.QLineEdit(Form)
        self.maxText.setObjectName("maxText")
        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)
        self.invertCheck = QtWidgets.QCheckBox(Form)
        self.invertCheck.setObjectName("invertCheck")
        self.gridLayout.addWidget(self.invertCheck, 5, 0, 1, 4)
        self.mouseCheck = QtWidgets.QCheckBox(Form)
        self.mouseCheck.setChecked(True)
        self.mouseCheck.setObjectName("mouseCheck")
        self.gridLayout.addWidget(self.mouseCheck, 6, 0, 1, 4)
        self.visibleOnlyCheck = QtWidgets.QCheckBox(Form)
        self.visibleOnlyCheck.setObjectName("visibleOnlyCheck")
        self.gridLayout.addWidget(self.visibleOnlyCheck, 3, 2, 1, 2)
        self.autoPanCheck = QtWidgets.QCheckBox(Form)
        self.autoPanCheck.setObjectName("autoPanCheck")
        self.gridLayout.addWidget(self.autoPanCheck, 4, 2, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Link Axis:", None, -1))
        self.linkCombo.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Links this axis with another view. When linked, both views will display the same data range.</p></body></html>", None, -1))
        self.autoPercentSpin.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Percent of data to be visible when auto-scaling. It may be useful to decrease this value for data with spiky noise.</p></body></html>", None, -1))
        self.autoPercentSpin.setSuffix(QtWidgets.QApplication.translate("Form", "%", None, -1))
        self.autoRadio.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None, -1))
        self.autoRadio.setText(QtWidgets.QApplication.translate("Form", "Auto", None, -1))
        self.manualRadio.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None, -1))
        self.manualRadio.setText(QtWidgets.QApplication.translate("Form", "Manual", None, -1))
        self.minText.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None, -1))
        self.minText.setText(QtWidgets.QApplication.translate("Form", "0", None, -1))
        self.maxText.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None, -1))
        self.maxText.setText(QtWidgets.QApplication.translate("Form", "0", None, -1))
        self.invertCheck.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Inverts the display of this axis. (+y points downward instead of upward)</p></body></html>", None, -1))
        self.invertCheck.setText(QtWidgets.QApplication.translate("Form", "Invert Axis", None, -1))
        self.mouseCheck.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>Enables mouse interaction (panning, scaling) for this axis.</p></body></html>", None, -1))
        self.mouseCheck.setText(QtWidgets.QApplication.translate("Form", "Mouse Enabled", None, -1))
        self.visibleOnlyCheck.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>When checked, the axis will only auto-scale to data that is visible along the orthogonal axis.</p></body></html>", None, -1))
        self.visibleOnlyCheck.setText(QtWidgets.QApplication.translate("Form", "Visible Data Only", None, -1))
        self.autoPanCheck.setToolTip(QtWidgets.QApplication.translate("Form", "<html><head/><body><p>When checked, the axis will automatically pan to center on the current data, but the scale along this axis will not change.</p></body></html>", None, -1))
        self.autoPanCheck.setText(QtWidgets.QApplication.translate("Form", "Auto Pan Only", None, -1))

