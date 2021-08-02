# -*- coding: utf-8 -*-
from ...Qt import QtCore, QtGui, QT_LIB
from ...WidgetGroup import WidgetGroup

import importlib
ui_template = importlib.import_module(
    f'.axisCtrlTemplate_{QT_LIB.lower()}', package=__package__)

import weakref 

translate = QtCore.QCoreApplication.translate
class ViewBoxMenu(QtGui.QMenu):
    def __init__(self, view):
        QtGui.QMenu.__init__(self)
        
        self.view = weakref.ref(view)  ## keep weakref to view to avoid circular reference (don't know why, but this prevents the ViewBox from being collected)
        self.valid = False  ## tells us whether the ui needs to be updated
        self.viewMap = weakref.WeakValueDictionary()  ## weakrefs to all views listed in the link combos

        self.setTitle(translate("ViewBox", "ViewBox options"))
        self.viewAll = QtGui.QAction(translate("ViewBox", "View All"), self)
        self.viewAll.triggered.connect(self.autoRange)
        self.addAction(self.viewAll)
        
        self.axes = []
        self.ctrl = []
        self.widgetGroups = []
        self.dv = QtGui.QDoubleValidator(self)
        for axis in 'XY':
            m = QtGui.QMenu()
            m.setTitle(f"{axis} {translate('ViewBox', 'axis')}")
            w = QtGui.QWidget()
            ui = ui_template.Ui_Form()
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
                (ui.minText.editingFinished, 'RangeTextChanged'),
                (ui.maxText.editingFinished, 'RangeTextChanged'),
                (ui.autoRadio.clicked, 'AutoClicked'),
                (ui.autoPercentSpin.valueChanged, 'AutoSpinChanged'),
                (ui.linkCombo.currentIndexChanged, 'LinkComboChanged'),
                (ui.autoPanCheck.toggled, 'AutoPanToggled'),
                (ui.visibleOnlyCheck.toggled, 'VisibleOnlyToggled')
            ]
            
            for sig, fn in connects:
                sig.connect(getattr(self, axis.lower()+fn))

        self.ctrl[0].invertCheck.toggled.connect(self.xInvertToggled)
        self.ctrl[1].invertCheck.toggled.connect(self.yInvertToggled)
        ## exporting is handled by GraphicsScene now
        #self.export = QtGui.QMenu("Export")
        #self.setExportMethods(view.exportMethods)
        #self.addMenu(self.export)
        
        self.leftMenu = QtGui.QMenu(translate("ViewBox", "Mouse Mode"))
        group = QtGui.QActionGroup(self)
        
        # This does not work! QAction _must_ be initialized with a permanent 
        # object as the parent or else it may be collected prematurely.
        #pan = self.leftMenu.addAction("3 button", self.set3ButtonMode)
        #zoom = self.leftMenu.addAction("1 button", self.set1ButtonMode)
        pan = QtGui.QAction(translate("ViewBox", "3 button"), self.leftMenu)
        zoom = QtGui.QAction(translate("ViewBox", "1 button"), self.leftMenu)
        self.leftMenu.addAction(pan)
        self.leftMenu.addAction(zoom)
        pan.triggered.connect(self.set3ButtonMode)
        zoom.triggered.connect(self.set1ButtonMode)
        
        pan.setCheckable(True)
        zoom.setCheckable(True)
        pan.setActionGroup(group)
        zoom.setActionGroup(group)
        self.mouseModes = [pan, zoom]
        self.addMenu(self.leftMenu)
        
        self.view().sigStateChanged.connect(self.viewStateChanged)
        
        self.updateState()

    def setExportMethods(self, methods):
        self.exportMethods = methods
        self.export.clear()
        for opt, fn in methods.items():
            self.export.addAction(opt, self.exportMethod)
        

    def viewStateChanged(self):
        self.valid = False
        if self.ctrl[0].minText.isVisible() or self.ctrl[1].minText.isVisible():
            self.updateState()
        
    def updateState(self):
        ## Something about the viewbox has changed; update the menu GUI
        
        state = self.view().getState(copy=False)
        if state['mouseMode'] == ViewBox.PanMode:
            self.mouseModes[0].setChecked(True)
        else:
            self.mouseModes[1].setChecked(True)
            
        for i in [0,1]:  # x, y
            tr = state['targetRange'][i]
            self.ctrl[i].minText.setText("%0.5g" % tr[0])
            self.ctrl[i].maxText.setText("%0.5g" % tr[1])
            if state['autoRange'][i] is not False:
                self.ctrl[i].autoRadio.setChecked(True)
                if state['autoRange'][i] is not True:
                    self.ctrl[i].autoPercentSpin.setValue(state['autoRange'][i]*100)
            else:
                self.ctrl[i].manualRadio.setChecked(True)
            self.ctrl[i].mouseCheck.setChecked(state['mouseEnabled'][i])
            
            ## Update combo to show currently linked view
            c = self.ctrl[i].linkCombo
            c.blockSignals(True)
            try:
                view = state['linkedViews'][i]  ## will always be string or None
                if view is None:
                    view = ''
                    
                ind = c.findText(view)
                    
                if ind == -1:
                    ind = 0
                c.setCurrentIndex(ind)
            finally:
                c.blockSignals(False)
            
            self.ctrl[i].autoPanCheck.setChecked(state['autoPan'][i])
            self.ctrl[i].visibleOnlyCheck.setChecked(state['autoVisibleOnly'][i])
            xy = ['x', 'y'][i]
            self.ctrl[i].invertCheck.setChecked(state.get(xy+'Inverted', False))
        
        self.valid = True
        
    def popup(self, *args):
        if not self.valid:
            self.updateState()
        QtGui.QMenu.popup(self, *args)
        
    def autoRange(self):
        self.view().autoRange()  ## don't let signal call this directly--it'll add an unwanted argument

    def xMouseToggled(self, b):
        self.view().setMouseEnabled(x=b)

    def xManualClicked(self):
        self.view().enableAutoRange(ViewBox.XAxis, False)
        
    def xRangeTextChanged(self):
        self.ctrl[0].manualRadio.setChecked(True)
        self.view().setXRange(*self._validateRangeText(0), padding=0)

    def xAutoClicked(self):
        val = self.ctrl[0].autoPercentSpin.value() * 0.01
        self.view().enableAutoRange(ViewBox.XAxis, val)
        
    def xAutoSpinChanged(self, val):
        self.ctrl[0].autoRadio.setChecked(True)
        self.view().enableAutoRange(ViewBox.XAxis, val*0.01)

    def xLinkComboChanged(self, ind):
        self.view().setXLink(str(self.ctrl[0].linkCombo.currentText()))

    def xAutoPanToggled(self, b):
        self.view().setAutoPan(x=b)
    
    def xVisibleOnlyToggled(self, b):
        self.view().setAutoVisible(x=b)


    def yMouseToggled(self, b):
        self.view().setMouseEnabled(y=b)

    def yManualClicked(self):
        self.view().enableAutoRange(ViewBox.YAxis, False)
        
    def yRangeTextChanged(self):
        self.ctrl[1].manualRadio.setChecked(True)
        self.view().setYRange(*self._validateRangeText(1), padding=0)
        
    def yAutoClicked(self):
        val = self.ctrl[1].autoPercentSpin.value() * 0.01
        self.view().enableAutoRange(ViewBox.YAxis, val)
        
    def yAutoSpinChanged(self, val):
        self.ctrl[1].autoRadio.setChecked(True)
        self.view().enableAutoRange(ViewBox.YAxis, val*0.01)

    def yLinkComboChanged(self, ind):
        self.view().setYLink(str(self.ctrl[1].linkCombo.currentText()))

    def yAutoPanToggled(self, b):
        self.view().setAutoPan(y=b)
    
    def yVisibleOnlyToggled(self, b):
        self.view().setAutoVisible(y=b)

    def yInvertToggled(self, b):
        self.view().invertY(b)

    def xInvertToggled(self, b):
        self.view().invertX(b)

    def exportMethod(self):
        act = self.sender()
        self.exportMethods[str(act.text())]()

    def set3ButtonMode(self):
        self.view().setLeftButtonAction('pan')
        
    def set1ButtonMode(self):
        self.view().setLeftButtonAction('rect')
        
    def setViewList(self, views):
        names = ['']
        self.viewMap.clear()
        
        ## generate list of views to show in the link combo
        for v in views:
            name = v.name
            if name is None:  ## unnamed views do not show up in the view list (although they are linkable)
                continue
            names.append(name)
            self.viewMap[name] = v
            
        for i in [0,1]:
            c = self.ctrl[i].linkCombo
            current = c.currentText()
            c.blockSignals(True)
            changed = True
            try:
                c.clear()
                for name in names:
                    c.addItem(name)
                    if name == current:
                        changed = False
                        c.setCurrentIndex(c.count()-1)
            finally:
                c.blockSignals(False)
                
            if changed:
                c.setCurrentIndex(0)
                c.currentIndexChanged.emit(c.currentIndex())

    def _validateRangeText(self, axis):
        """Validate range text inputs. Return current value(s) if invalid."""
        inputs = (self.ctrl[axis].minText.text(),
                  self.ctrl[axis].maxText.text())
        vals = self.view().viewRange()[axis]
        for i, text in enumerate(inputs):
            try:
                vals[i] = float(text)
            except ValueError:
                # could not convert string to float
                pass
        return vals

        
from .ViewBox import ViewBox
        
    
