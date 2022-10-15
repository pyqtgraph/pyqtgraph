import sys, re, traceback
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget


class ExceptionHandlerWidget(QtWidgets.QGroupBox):
    sigStackItemClicked = QtCore.Signal(object, object)  # self, item
    sigStackItemDblClicked = QtCore.Signal(object, object)  # self, item
    _threadException = QtCore.Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setupUi()

        # send exceptions raised in non-gui threads back to the main thread by signal.
        self._threadException.connect(self._threadExceptionHandler)

    def _setupUi(self):
        self.setTitle("Exception Handling")

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setHorizontalSpacing(2)
        self.layout.setVerticalSpacing(0)

        self.clearExceptionBtn = QtWidgets.QPushButton("Clear Stack", self)
        self.clearExceptionBtn.setEnabled(False)
        self.layout.addWidget(self.clearExceptionBtn, 0, 6, 1, 1)

        self.catchAllExceptionsBtn = QtWidgets.QPushButton("Show All Exceptions", self)
        self.catchAllExceptionsBtn.setCheckable(True)
        self.layout.addWidget(self.catchAllExceptionsBtn, 0, 1, 1, 1)

        self.catchNextExceptionBtn = QtWidgets.QPushButton("Show Next Exception", self)
        self.catchNextExceptionBtn.setCheckable(True)
        self.layout.addWidget(self.catchNextExceptionBtn, 0, 0, 1, 1)

        self.onlyUncaughtCheck = QtWidgets.QCheckBox("Only Uncaught Exceptions", self)
        self.onlyUncaughtCheck.setChecked(True)
        self.layout.addWidget(self.onlyUncaughtCheck, 0, 4, 1, 1)

        self.exceptionStackList = StackWidget(self)
        self.layout.addWidget(self.exceptionStackList, 2, 0, 1, 7)

        self.runSelectedFrameCheck = QtWidgets.QCheckBox("Run commands in selected stack frame", self)
        self.runSelectedFrameCheck.setChecked(True)
        self.layout.addWidget(self.runSelectedFrameCheck, 3, 0, 1, 7)

        self.exceptionInfoLabel = QtWidgets.QLabel("Stack Trace", self)
        self.exceptionInfoLabel.setWordWrap(True)
        self.layout.addWidget(self.exceptionInfoLabel, 1, 0, 1, 7)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout.addItem(spacerItem, 0, 5, 1, 1)

        self.filterLabel = QtWidgets.QLabel("Filter (regex):", self)
        self.layout.addWidget(self.filterLabel, 0, 2, 1, 1)

        self.filterText = QtWidgets.QLineEdit(self)
        self.layout.addWidget(self.filterText, 0, 3, 1, 1)

        self.catchAllExceptionsBtn.toggled.connect(self.catchAllExceptions)
        self.catchNextExceptionBtn.toggled.connect(self.catchNextException)
        self.clearExceptionBtn.clicked.connect(self.clearExceptionClicked)
        self.exceptionStackList.itemClicked.connect(self.stackItemClicked)
        self.exceptionStackList.itemDoubleClicked.connect(self.stackItemDblClicked)
        self.onlyUncaughtCheck.toggled.connect(self.updateSysTrace)

    def setStack(self, frame=None, tb=None):
        self.clearExceptionBtn.setEnabled(True)
        self.exceptionStackList.setStack(frame, tb)

    def catchAllExceptions(self, catch=True):
        """
        If True, the console will catch all unhandled exceptions and display the stack
        trace. Each exception caught clears the last.
        """
        with SignalBlock(self.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
            self.catchAllExceptionsBtn.setChecked(catch)
        
        if catch:
            with SignalBlock(self.catchNextExceptionBtn.toggled, self.catchNextException):
                self.catchNextExceptionBtn.setChecked(False)
            self.enableExceptionHandling()
        else:
            self.disableExceptionHandling()
        
    def catchNextException(self, catch=True):
        """
        If True, the console will catch the next unhandled exception and display the stack
        trace.
        """
        with SignalBlock(self.catchNextExceptionBtn.toggled, self.catchNextException):
            self.catchNextExceptionBtn.setChecked(catch)
        if catch:
            with SignalBlock(self.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
                self.catchAllExceptionsBtn.setChecked(False)
            self.enableExceptionHandling()
        else:
            self.disableExceptionHandling()
        
    def enableExceptionHandling(self):
        exceptionHandling.register(self.exceptionHandler)
        self.updateSysTrace()
        
    def disableExceptionHandling(self):
        exceptionHandling.unregister(self.exceptionHandler)
        self.updateSysTrace()
        
    def clearExceptionClicked(self):
        self.currentTraceback = None
        self.exceptionInfoLabel.setText("[No current exception]")
        self.exceptionStackList.clear()
        self.clearExceptionBtn.setEnabled(False)
        
    def updateSysTrace(self):
        ## Install or uninstall  sys.settrace handler 
        
        if not self.catchNextExceptionBtn.isChecked() and not self.catchAllExceptionsBtn.isChecked():
            if sys.gettrace() == self.systrace:
                sys.settrace(None)
            return
        
        if self.onlyUncaughtCheck.isChecked():
            if sys.gettrace() == self.systrace:
                sys.settrace(None)
        else:
            if sys.gettrace() is not None and sys.gettrace() != self.systrace:
                self.onlyUncaughtCheck.setChecked(False)
                raise Exception("sys.settrace is in use; cannot monitor for caught exceptions.")
            else:
                sys.settrace(self.systrace)
        
    def exceptionHandler(self, excType, exc, tb, systrace=False, frame=None):
        if frame is None:
            frame = sys._getframe()

        # exceptions raised in non-gui threads must be handled separately
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            # sending a frame from one thread to another.. probably not safe, but better than just
            # dropping the exception?
            self._threadException.emit((excType, exc, tb, systrace, frame.f_back))
            return

        if self.catchNextExceptionBtn.isChecked():
            self.catchNextExceptionBtn.setChecked(False)
        elif not self.catchAllExceptionsBtn.isChecked():
            return
        
        self.currentTraceback = tb
        
        excMessage = ''.join(traceback.format_exception_only(excType, exc))
        self.exceptionInfoLabel.setText(excMessage)

        if systrace:
            # exceptions caught using systrace don't need the usual 
            # call stack + traceback handling
            self.setStack(frame.f_back.f_back)
        else:
            self.setStack(frame=frame.f_back, tb=tb)
    
    def _threadExceptionHandler(self, args):
        self.exceptionHandler(*args)

    def systrace(self, frame, event, arg):
        if event == 'exception' and self.checkException(*arg):
            self.exceptionHandler(*arg, systrace=True)
        return self.systrace
        
    def checkException(self, excType, exc, tb):
        ## Return True if the exception is interesting; False if it should be ignored.
        
        filename = tb.tb_frame.f_code.co_filename
        function = tb.tb_frame.f_code.co_name
        
        filterStr = str(self.filterText.text())
        if filterStr != '':
            if isinstance(exc, Exception):
                msg = traceback.format_exception_only(type(exc), exc)
            elif isinstance(exc, str):
                msg = exc
            else:
                msg = repr(exc)
            match = re.search(filterStr, "%s:%s:%s" % (filename, function, msg))
            return match is not None

        ## Go through a list of common exception points we like to ignore:
        if excType is GeneratorExit or excType is StopIteration:
            return False
        if excType is AttributeError:
            if filename.endswith('numpy/core/fromnumeric.py') and function in ('all', '_wrapit', 'transpose', 'sum'):
                return False
            if filename.endswith('numpy/core/arrayprint.py') and function in ('_array2string'):
                return False
            if filename.endswith('MetaArray.py') and function == '__getattr__':
                for name in ('__array_interface__', '__array_struct__', '__array__'):  ## numpy looks for these when converting objects to array
                    if name in exc:
                        return False
            if filename.endswith('flowchart/eq.py'):
                return False
        if excType is TypeError:
            if filename.endswith('numpy/lib/function_base.py') and function == 'iterable':
                return False
            
        return True
    
    def stackItemClicked(self, item):
        self.sigStackItemClicked.emit(self, item)

    def stackItemDblClicked(self, item):
        self.sigStackItemDblClicked.emit(self, item)
