# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore

__all__ = ['ProgressDialog']


class ProgressDialog(QtGui.QProgressDialog):
    """
    Extends QProgressDialog for use in 'with' statements.

    Example::

        with ProgressDialog("Processing..", minVal, maxVal) as dlg:
            # do stuff
            dlg.setValue(i)   ## could also use dlg += 1
            if dlg.wasCanceled():
                raise Exception("Processing canceled by user")
    """
    
    allDialogs = []
    
    def __init__(self, labelText, minimum=0, maximum=100, cancelText='Cancel', parent=None, wait=250, busyCursor=False, disable=False, nested=False):
        """
        ============== ================================================================
        **Arguments:**
        labelText      (required)
        cancelText     Text to display on cancel button, or None to disable it.
        minimum
        maximum
        parent       
        wait           Length of time (im ms) to wait before displaying dialog
        busyCursor     If True, show busy cursor until dialog finishes
        disable        If True, the progress dialog will not be displayed
                       and calls to wasCanceled() will always return False.
                       If ProgressDialog is entered from a non-gui thread, it will
                       always be disabled.
        nested         (bool) If True, then this progress bar will be displayed inside
                       any pre-existing progress dialogs that also allow nesting (if
                       any).
        ============== ================================================================
        """    
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        self.disabled = disable or (not isGuiThread)
        if self.disabled:
            return

        noCancel = False
        if cancelText is None:
            cancelText = ''
            noCancel = True
            
        self.busyCursor = busyCursor

        QtGui.QProgressDialog.__init__(self, labelText, cancelText, minimum, maximum, parent)
        
        # If this will be a nested dialog, then we ignore the wait time
        if nested is True and len(ProgressDialog.allDialogs) > 0:
            self.setMinimumDuration(2**30)
        else:
            self.setMinimumDuration(wait)
            
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setValue(self.minimum())
        if noCancel:
            self.setCancelButton(None)

        # attributes used for nesting dialogs
        self.nestedLayout = None
        self._nestableWidgets = None
        self._nestingReady = False
        self._topDialog = None
        
    def __enter__(self):
        if self.disabled:
            return self
        if self.busyCursor:
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        
        if len(ProgressDialog.allDialogs) > 0:
            topDialog = ProgressDialog.allDialogs[0]
            topDialog._addSubDialog(self)
            self._topDialog = topDialog
            topDialog.canceled.connect(self.cancel)
        
        ProgressDialog.allDialogs.append(self)
        
        return self

    def __exit__(self, exType, exValue, exTrace):
        if self.disabled:
            return
        if self.busyCursor:
            QtGui.QApplication.restoreOverrideCursor()
            
        if self._topDialog is not None:
            self._topDialog._removeSubDialog(self)
        
        ProgressDialog.allDialogs.pop(-1)

        self.setValue(self.maximum())
        
    def __iadd__(self, val):
        """Use inplace-addition operator for easy incrementing."""
        if self.disabled:
            return self
        self.setValue(self.value()+val)
        return self

    def _addSubDialog(self, dlg):
        # insert widgets from another dialog into this one.
        
        # set a new layout and arrange children into it (if needed).
        self._prepareNesting()
        
        bar, btn = dlg._extractWidgets()
        bar.removed = False
        
        # where should we insert this widget? Find the first slot with a 
        # "removed" widget (that was left as a placeholder)
        nw = self.nestedLayout.count()
        inserted = False
        if nw > 1:
            for i in range(1, nw):
                bar2 = self.nestedLayout.itemAt(i).widget()
                if bar2.removed:
                    self.nestedLayout.removeWidget(bar2)
                    bar2.hide()
                    bar2.setParent(None)
                    self.nestedLayout.insertWidget(i, bar)
                    inserted = True
                    break
        if not inserted:
            self.nestedLayout.addWidget(bar)

    def _removeSubDialog(self, dlg):
        # don't remove the widget just yet; instead we hide it and leave it in 
        # as a placeholder.
        bar, btn = dlg._extractWidgets()
        bar.layout().setCurrentIndex(1)  # causes widgets to be hidden without changing size
        bar.removed = True # mark as removed so we know we can insert another bar here later

    def _prepareNesting(self):
        # extract all child widgets and place into a new layout that we can add to
        if self._nestingReady is False:
            # top layout contains progress bars + cancel button at the bottom
            self._topLayout = QtGui.QGridLayout()
            self.setLayout(self._topLayout)
            self._topLayout.setContentsMargins(0, 0, 0, 0)
            
            # A vbox to contain all progress bars
            self.nestedVBox = QtGui.QWidget()
            self._topLayout.addWidget(self.nestedVBox, 0, 0, 1, 2)
            self.nestedLayout = QtGui.QVBoxLayout()
            self.nestedVBox.setLayout(self.nestedLayout)
            
            # re-insert all widgets
            bar, btn = self._extractWidgets()
            self.nestedLayout.addWidget(bar)
            self._topLayout.addWidget(btn, 1, 1, 1, 1)
            self._topLayout.setColumnStretch(0, 100)
            self._topLayout.setColumnStretch(1, 1)
            self._topLayout.setRowStretch(0, 100)
            self._topLayout.setRowStretch(1, 1)
            
            self._nestingReady = True

    def _extractWidgets(self):
        # return a single widget containing all sub-widgets nicely arranged
        if self._nestableWidgets is None:
            widgets = [ch for ch in self.children() if isinstance(ch, QtGui.QWidget)]
            label = [ch for ch in self.children() if isinstance(ch, QtGui.QLabel)][0]
            bar = [ch for ch in self.children() if isinstance(ch, QtGui.QProgressBar)][0]
            btn = [ch for ch in self.children() if isinstance(ch, QtGui.QPushButton)][0]
            
            # join label and bar into a stacked layout so they can be hidden
            # without changing size
            sw = QtGui.QWidget()
            sl = QtGui.QStackedLayout()
            sw.setLayout(sl)
            sl.setContentsMargins(0, 0, 0, 0)
            
            # inside the stacked layout, the bar and label are in a vbox
            w = QtGui.QWidget()
            sl.addWidget(w)
            l = QtGui.QVBoxLayout()
            w.setLayout(l)
            l.addWidget(label)
            l.addWidget(bar)
            
            # add a blank page to the stacked layout
            blank = QtGui.QWidget()
            sl.addWidget(blank)
            
            self._nestableWidgets = (sw, btn)
            
        return self._nestableWidgets

    ## wrap all other functions to make sure they aren't being called from non-gui threads
    
    def setValue(self, val):
        if self.disabled:
            return
        QtGui.QProgressDialog.setValue(self, val)
        
        # Qt docs say this should happen automatically, but that doesn't seem
        # to be the case.
        if self.windowModality() == QtCore.Qt.WindowModal:
            QtGui.QApplication.processEvents()
        
    def setLabelText(self, val):
        if self.disabled:
            return
        QtGui.QProgressDialog.setLabelText(self, val)
    
    def setMaximum(self, val):
        if self.disabled:
            return
        QtGui.QProgressDialog.setMaximum(self, val)

    def setMinimum(self, val):
        if self.disabled:
            return
        QtGui.QProgressDialog.setMinimum(self, val)
        
    def wasCanceled(self):
        if self.disabled:
            return False
        return QtGui.QProgressDialog.wasCanceled(self)

    def maximum(self):
        if self.disabled:
            return 0
        return QtGui.QProgressDialog.maximum(self)

    def minimum(self):
        if self.disabled:
            return 0
        return QtGui.QProgressDialog.minimum(self)
