# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './pyqtgraph/graphicsItems/ViewBox/axisCtrlTemplate.ui'
#
# Created: Mon Dec 23 10:10:51 2013
#      by: PyQt4 UI code generator 4.10
#
# WARNING! All changes made in this file will be lost!

from ...Qt import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(186, 154)
        Form.setMaximumSize(QtCore.QSize(200, 16777215))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 7, 0, 1, 2)
        self.linkCombo = QtGui.QComboBox(Form)
        self.linkCombo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)
        self.linkCombo.setObjectName(_fromUtf8("linkCombo"))
        self.gridLayout.addWidget(self.linkCombo, 7, 2, 1, 2)
        self.autoPercentSpin = QtGui.QSpinBox(Form)
        self.autoPercentSpin.setEnabled(True)
        self.autoPercentSpin.setMinimum(1)
        self.autoPercentSpin.setMaximum(100)
        self.autoPercentSpin.setSingleStep(1)
        self.autoPercentSpin.setProperty("value", 100)
        self.autoPercentSpin.setObjectName(_fromUtf8("autoPercentSpin"))
        self.gridLayout.addWidget(self.autoPercentSpin, 2, 2, 1, 2)
        self.autoRadio = QtGui.QRadioButton(Form)
        self.autoRadio.setChecked(True)
        self.autoRadio.setObjectName(_fromUtf8("autoRadio"))
        self.gridLayout.addWidget(self.autoRadio, 2, 0, 1, 2)
        self.manualRadio = QtGui.QRadioButton(Form)
        self.manualRadio.setObjectName(_fromUtf8("manualRadio"))
        self.gridLayout.addWidget(self.manualRadio, 1, 0, 1, 2)
        self.minText = QtGui.QLineEdit(Form)
        self.minText.setObjectName(_fromUtf8("minText"))
        self.gridLayout.addWidget(self.minText, 1, 2, 1, 1)
        self.maxText = QtGui.QLineEdit(Form)
        self.maxText.setObjectName(_fromUtf8("maxText"))
        self.gridLayout.addWidget(self.maxText, 1, 3, 1, 1)
        self.invertCheck = QtGui.QCheckBox(Form)
        self.invertCheck.setObjectName(_fromUtf8("invertCheck"))
        self.gridLayout.addWidget(self.invertCheck, 5, 0, 1, 4)
        self.mouseCheck = QtGui.QCheckBox(Form)
        self.mouseCheck.setChecked(True)
        self.mouseCheck.setObjectName(_fromUtf8("mouseCheck"))
        self.gridLayout.addWidget(self.mouseCheck, 6, 0, 1, 4)
        self.visibleOnlyCheck = QtGui.QCheckBox(Form)
        self.visibleOnlyCheck.setObjectName(_fromUtf8("visibleOnlyCheck"))
        self.gridLayout.addWidget(self.visibleOnlyCheck, 3, 2, 1, 2)
        self.autoPanCheck = QtGui.QCheckBox(Form)
        self.autoPanCheck.setObjectName(_fromUtf8("autoPanCheck"))
        self.gridLayout.addWidget(self.autoPanCheck, 4, 2, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "Link Axis:", None))
        self.linkCombo.setToolTip(_translate("Form", "<html><head/><body><p>Links this axis with another view. When linked, both views will display the same data range.</p></body></html>", None))
        self.autoPercentSpin.setToolTip(_translate("Form", "<html><head/><body><p>Percent of data to be visible when auto-scaling. It may be useful to decrease this value for data with spiky noise.</p></body></html>", None))
        self.autoPercentSpin.setSuffix(_translate("Form", "%", None))
        self.autoRadio.setToolTip(_translate("Form", "<html><head/><body><p>Automatically resize this axis whenever the displayed data is changed.</p></body></html>", None))
        self.autoRadio.setText(_translate("Form", "Auto", None))
        self.manualRadio.setToolTip(_translate("Form", "<html><head/><body><p>Set the range for this axis manually. This disables automatic scaling. </p></body></html>", None))
        self.manualRadio.setText(_translate("Form", "Manual", None))
        self.minText.setToolTip(_translate("Form", "<html><head/><body><p>Minimum value to display for this axis.</p></body></html>", None))
        self.minText.setText(_translate("Form", "0", None))
        self.maxText.setToolTip(_translate("Form", "<html><head/><body><p>Maximum value to display for this axis.</p></body></html>", None))
        self.maxText.setText(_translate("Form", "0", None))
        self.invertCheck.setToolTip(_translate("Form", "<html><head/><body><p>Inverts the display of this axis. (+y points downward instead of upward)</p></body></html>", None))
        self.invertCheck.setText(_translate("Form", "Invert Axis", None))
        self.mouseCheck.setToolTip(_translate("Form", "<html><head/><body><p>Enables mouse interaction (panning, scaling) for this axis.</p></body></html>", None))
        self.mouseCheck.setText(_translate("Form", "Mouse Enabled", None))
        self.visibleOnlyCheck.setToolTip(_translate("Form", "<html><head/><body><p>When checked, the axis will only auto-scale to data that is visible along the orthogonal axis.</p></body></html>", None))
        self.visibleOnlyCheck.setText(_translate("Form", "Visible Data Only", None))
        self.autoPanCheck.setToolTip(_translate("Form", "<html><head/><body><p>When checked, the axis will automatically pan to center on the current data, but the scale along this axis will not change.</p></body></html>", None))
        self.autoPanCheck.setText(_translate("Form", "Auto Pan Only", None))

