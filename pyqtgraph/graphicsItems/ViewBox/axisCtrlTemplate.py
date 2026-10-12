from ...Qt import QtCore, QtWidgets

translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(186, 154)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        Form.setWindowTitle(translate("Form", "PyQtGraph"))

        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.manualRadio = QtWidgets.QRadioButton(Form)
        self.manualRadio.setObjectName("manualRadio")
        self.manualRadio.setToolTip(translate("Form", "<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>"))
        self.manualRadio.setText(translate("Form", "Manual"))
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)

        self.minText = QtWidgets.QLineEdit(Form)
        self.minText.setObjectName("minText")
        self.minText.setToolTip(translate("Form", "<html><head/><body><p>Minimum value to display for this axis.</p></body></html>"))
        self.minText.setText(translate("Form", "0"))
        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)

        self.maxText = QtWidgets.QLineEdit(Form)
        self.maxText.setObjectName("maxText")
        self.maxText.setToolTip(translate("Form", "<html><head/><body><p>Maximum value to display for this axis.</p></body></html>"))
        self.maxText.setText(translate("Form", "0"))
        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)

        self.autoRadio = QtWidgets.QRadioButton(Form)
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName("autoRadio")
        self.autoRadio.setToolTip(translate("Form", "<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>"))
        self.autoRadio.setText(translate("Form", "Auto"))
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)

        self.autoPercentSpin = QtWidgets.QSpinBox(Form)
        self.autoPercentSpin.setEnabled(True)
        self.autoPercentSpin.setMinimum(1)
        self.autoPercentSpin.setMaximum(100)
        self.autoPercentSpin.setSingleStep(1)
        self.autoPercentSpin.setProperty("value", 100)
        self.autoPercentSpin.setObjectName("autoPercentSpin")
        self.autoPercentSpin.setToolTip(translate("Form", "<html><head/><body><p>Percent of data to be visible when auto-scaling. It may be useful to decrease this value for data with spiky noise.</p></body></html>"))
        self.autoPercentSpin.setSuffix(translate("Form", "%"))
        self.gridLayout.addWidget(self.autoPercentSpin, 2, 2, 1, 2)

        self.visibleOnlyCheck = QtWidgets.QCheckBox(Form)
        self.visibleOnlyCheck.setObjectName("visibleOnlyCheck")
        self.visibleOnlyCheck.setToolTip(translate("Form", "<html><head/><body><p>When checked, the axis will only auto-scale to data that is visible along the orthogonal axis.</p></body></html>"))
        self.visibleOnlyCheck.setText(translate("Form", "Visible Data Only"))
        self.gridLayout.addWidget(self.visibleOnlyCheck, 3, 2, 1, 2)

        self.autoPanCheck = QtWidgets.QCheckBox(Form)
        self.autoPanCheck.setObjectName("autoPanCheck")
        self.autoPanCheck.setToolTip(translate("Form", "<html><head/><body><p>When checked, the axis will automatically pan to center on the current data, but the scale along this axis will not change.</p></body></html>"))
        self.autoPanCheck.setText(translate("Form", "Auto Pan Only"))
        self.gridLayout.addWidget(self.autoPanCheck, 4, 2, 1, 2)

        self.invertCheck = QtWidgets.QCheckBox(Form)
        self.invertCheck.setObjectName("invertCheck")
        self.invertCheck.setToolTip(translate("Form", "<html><head/><body><p>Inverts the display of this axis. (+y points downward instead of upward)</p></body></html>"))
        self.invertCheck.setText(translate("Form", "Invert Axis"))
        self.gridLayout.addWidget(self.invertCheck, 5, 0, 1, 4)

        self.mouseCheck = QtWidgets.QCheckBox(Form)
        self.mouseCheck.setChecked(True)
        self.mouseCheck.setObjectName("mouseCheck")
        self.mouseCheck.setToolTip(translate("Form", "<html><head/><body><p>Enables mouse interaction (panning, scaling) for this axis.</p></body></html>"))
        self.mouseCheck.setText(translate("Form", "Mouse Enabled"))
        self.gridLayout.addWidget(self.mouseCheck, 6, 0, 1, 4)

        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.label.setText(translate("Form", "Link Axis:"))
        self.gridLayout.addWidget(self.label, 7, 0, 1, 2)

        self.linkCombo = QtWidgets.QComboBox(Form)
        self.linkCombo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.linkCombo.setObjectName("linkCombo")
        self.linkCombo.setToolTip(translate("Form", "<html><head/><body><p>Links this axis with another view. When linked, both views will display the same data range.</p></body></html>"))
        self.gridLayout.addWidget(self.linkCombo, 7, 2, 1, 2)

        QtCore.QMetaObject.connectSlotsByName(Form)
