from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.WidgetGroup import WidgetGroup
from axisCtrlTemplate import Ui_Form as AxisCtrlTemplate

class ViewBoxMenu(QtGui.QMenu):
    def __init__(self, view):
        QtGui.QMenu.__init__(self)
        
        self.view = view
        self.valid = False  ## tells us whether the ui needs to be updated

        self.setTitle("ViewBox options")
        self.viewAll = QtGui.QAction("View All", self)
        self.viewAll.triggered.connect(self.autoRange)
        self.addAction(self.viewAll)
        
        self.axes = []
        self.ctrl = []
        self.widgetGroups = []
        self.dv = QtGui.QDoubleValidator(self)
        for axis in 'XY':
            m = QtGui.QMenu()
            m.setTitle("%s Axis" % axis)
            w = QtGui.QWidget()
            ui = AxisCtrlTemplate()
            ui.setupUi(w)
            a = QtGui.QWidgetAction(self)
            a.setDefaultWidget(w)
            m.addAction(a)
            self.addMenu(m)
            self.axes.append(m)
            self.ctrl.append(ui)
            wg = WidgetGroup(w)
            self.widgetGroups.append(w)
            
            connects = [
                (ui.mouseCheck.toggled, 'MouseToggled'),
                (ui.manualRadio.clicked, 'ManualClicked'),
                (ui.minText.editingFinished, 'MinTextChanged'),
                (ui.maxText.editingFinished, 'MaxTextChanged'),
                (ui.autoRadio.clicked, 'AutoClicked'),
                (ui.autoPercentSpin.valueChanged, 'AutoSpinChanged'),
                (ui.linkCombo.currentIndexChanged, 'LinkComboChanged'),
            ]
            
            for sig, fn in connects:
                sig.connect(getattr(self, axis.lower()+fn))
            
        self.export = QtGui.QMenu("Export")
        self.setExportMethods(view.exportMethods)
        self.addMenu(self.export)
        
        self.leftMenu = QtGui.QMenu("Mouse Mode")
        group = QtGui.QActionGroup(self)
        pan = self.leftMenu.addAction("3 button", self.set3ButtonMode)
        zoom = self.leftMenu.addAction("1 button", self.set1ButtonMode)
        pan.setCheckable(True)
        zoom.setCheckable(True)
        pan.setActionGroup(group)
        zoom.setActionGroup(group)
        self.mouseModes = [pan, zoom]
        self.addMenu(self.leftMenu)
        
        self.view.sigStateChanged.connect(self.viewStateChanged)
        
        self.updateState()

    def copy(self):
        m = QtGui.QMenu()
        for sm in self.subMenus():
            if isinstance(sm, QtGui.QMenu):
                m.addMenu(sm)
            else:
                m.addAction(sm)
        m.setTitle(self.title())
        return m

    def subMenus(self):
        if not self.valid:
            self.updateState()
        return [self.viewAll] + self.axes + [self.export, self.leftMenu]


    def setExportMethods(self, methods):
        self.exportMethods = methods
        self.export.clear()
        for opt, fn in methods.iteritems():
            self.export.addAction(opt, self.exportMethod)
        

    def viewStateChanged(self):
        self.valid = False
        if self.ctrl[0].minText.isVisible() or self.ctrl[1].minText.isVisible():
            self.updateState()
        
    def updateState(self):
        state = self.view.getState(copy=False)
        if state['mouseMode'] == ViewBox.PanMode:
            self.mouseModes[0].setChecked(True)
        else:
            self.mouseModes[1].setChecked(True)
            
            
        for i in [0,1]:
            tr = state['targetRange'][i]
            self.ctrl[i].minText.setText("%0.5g" % tr[0])
            self.ctrl[i].maxText.setText("%0.5g" % tr[1])
            if state['autoRange'][i] is not False:
                self.ctrl[i].autoRadio.setChecked(True)
            else:
                self.ctrl[i].manualRadio.setChecked(True)
            self.ctrl[i].mouseCheck.setChecked(state['mouseEnabled'][i])
            
            c = self.ctrl[i].linkCombo
            c.blockSignals(True)
            try:
                view = state['linkedViews'][i]
                if view is None:
                    view = ''
                ind = c.findText(view)
                if ind == -1:
                    ind = 0
                c.setCurrentIndex(ind)
            finally:
                c.blockSignals(False)
            
            
        self.valid = True
        
        
    def autoRange(self):
        self.view.autoRange()  ## don't let signal call this directly--it'll add an unwanted argument

    def xMouseToggled(self, b):
        self.view.setMouseEnabled(x=b)

    def xManualClicked(self):
        self.view.enableAutoRange(ViewBox.XAxis, False)
        
    def xMinTextChanged(self):
        self.ctrl[0].manualRadio.setChecked(True)
        self.view.setXRange(float(self.ctrl[0].minText.text()), float(self.ctrl[0].maxText.text()), padding=0)

    def xMaxTextChanged(self):
        self.ctrl[0].manualRadio.setChecked(True)
        self.view.setXRange(float(self.ctrl[0].minText.text()), float(self.ctrl[0].maxText.text()), padding=0)
        
    def xAutoClicked(self):
        val = self.ctrl[0].autoPercentSpin.value() * 0.01
        self.view.enableAutoRange(ViewBox.XAxis, val)
        
    def xAutoSpinChanged(self, val):
        self.ctrl[0].autoRadio.setChecked(True)
        self.view.enableAutoRange(ViewBox.XAxis, val*0.01)

    def xLinkComboChanged(self, ind):
        self.view.setXLink(str(self.ctrl[0].linkCombo.currentText()))



    def yMouseToggled(self, b):
        self.view.setMouseEnabled(y=b)

    def yManualClicked(self):
        self.view.enableAutoRange(ViewBox.YAxis, False)
        
    def yMinTextChanged(self):
        self.ctrl[1].manualRadio.setChecked(True)
        self.view.setYRange(float(self.ctrl[1].minText.text()), float(self.ctrl[1].maxText.text()), padding=0)
        
    def yMaxTextChanged(self):
        self.ctrl[1].manualRadio.setChecked(True)
        self.view.setYRange(float(self.ctrl[1].minText.text()), float(self.ctrl[1].maxText.text()), padding=0)
        
    def yAutoClicked(self):
        val = self.ctrl[1].autoPercentSpin.value() * 0.01
        self.view.enableAutoRange(ViewBox.YAxis, val)
        
    def yAutoSpinChanged(self, val):
        self.ctrl[1].autoRadio.setChecked(True)
        self.view.enableAutoRange(ViewBox.YAxis, val*0.01)

    def yLinkComboChanged(self, ind):
        self.view.setYLink(str(self.ctrl[1].linkCombo.currentText()))


    def exportMethod(self):
        act = self.sender()
        self.exportMethods[str(act.text())]()


    def set3ButtonMode(self):
        self.view.setLeftButtonAction('pan')
        
    def set1ButtonMode(self):
        self.view.setLeftButtonAction('rect')
        
        
    def setViewList(self, views):
        views = [''] + views
        for i in [0,1]:
            c = self.ctrl[i].linkCombo
            current = unicode(c.currentText())
            c.blockSignals(True)
            changed = True
            try:
                c.clear()
                for v in views:
                    c.addItem(v)
                    if v == current:
                        changed = False
                        c.setCurrentIndex(c.count()-1)
            finally:
                c.blockSignals(False)
                
            if changed:
                c.setCurrentIndex(0)
                c.currentIndexChanged.emit(c.currentIndex())
        
from ViewBox import ViewBox
        
    