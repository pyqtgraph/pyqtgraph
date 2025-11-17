import sys, re, traceback, threading
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

        self.filterString = ''
        self._inSystrace = False

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

        self.stackTree = StackWidget(self)
        self.layout.addWidget(self.stackTree, 2, 0, 1, 7)

        self.runSelectedFrameCheck = QtWidgets.QCheckBox("Run commands in selected stack frame", self)
        self.runSelectedFrameCheck.setChecked(True)
        self.layout.addWidget(self.runSelectedFrameCheck, 3, 0, 1, 7)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.layout.addItem(spacerItem, 0, 5, 1, 1)

        self.filterLabel = QtWidgets.QLabel("Filter (regex):", self)
        self.layout.addWidget(self.filterLabel, 0, 2, 1, 1)

        self.filterText = QtWidgets.QLineEdit(self)
        self.layout.addWidget(self.filterText, 0, 3, 1, 1)

        self.catchAllExceptionsBtn.toggled.connect(self.catchAllExceptions)
        self.catchNextExceptionBtn.toggled.connect(self.catchNextException)
        self.clearExceptionBtn.clicked.connect(self.clearExceptionClicked)
        self.stackTree.itemClicked.connect(self.stackItemClicked)
        self.stackTree.itemDoubleClicked.connect(self.stackItemDblClicked)
        self.onlyUncaughtCheck.toggled.connect(self.updateSysTrace)
        self.filterText.textChanged.connect(self._filterTextChanged)

    def setStack(self, frame=None):
        self.clearExceptionBtn.setEnabled(True)
        self.stackTree.setStack(frame)

    def setException(self, exc=None, lastFrame=None):
        self.clearExceptionBtn.setEnabled(True)
        self.stackTree.setException(exc, lastFrame=lastFrame)

    def selectedFrame(self):
        return self.stackTree.selectedFrame()

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
        exceptionHandling.registerCallback(self.exceptionHandler)
        self.updateSysTrace()
        
    def disableExceptionHandling(self):
        exceptionHandling.unregisterCallback(self.exceptionHandler)
        self.updateSysTrace()
        
    def clearExceptionClicked(self):
        self.stackTree.clear()
        self.clearExceptionBtn.setEnabled(False)
        
    def updateSysTrace(self):
        ## Install or uninstall  sys.settrace handler 
        
        if not self.catchNextExceptionBtn.isChecked() and not self.catchAllExceptionsBtn.isChecked():
            if sys.gettrace() == self.systrace:
                self._disableSysTrace()
            return
        
        if self.onlyUncaughtCheck.isChecked():
            if sys.gettrace() == self.systrace:
                self._disableSysTrace()
        else:
            if sys.gettrace() not in (None, self.systrace):
                self.onlyUncaughtCheck.setChecked(False)
                raise Exception("sys.settrace is in use (are you using another debugger?); cannot monitor for caught exceptions.")
            else:
                self._enableSysTrace()

    def _enableSysTrace(self):
        # set global trace function
        # note: this has no effect on pre-existing frames or threads 
        # until settrace_all_threads arrives in python 3.12.
        sys.settrace(self.systrace)  # affects current thread only
        threading.settrace(self.systrace)  # affects new threads only
        if hasattr(threading, 'settrace_all_threads'):
            threading.settrace_all_threads(self.systrace)

    def _disableSysTrace(self):
        sys.settrace(None)
        threading.settrace(None)
        if hasattr(threading, 'settrace_all_threads'):
            threading.settrace_all_threads(None)

    def exceptionHandler(self, excInfo, lastFrame=None):
        if isinstance(excInfo, Exception):
            exc = excInfo
        else:
            exc = excInfo.exc_value

        # exceptions raised in non-gui threads must be sent to the gui thread by signal
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            # note: we are giving the user the ability to modify a frame owned by another thread.. 
            # expect trouble :)
            self._threadException.emit((excInfo, lastFrame))
            return

        if self.catchNextExceptionBtn.isChecked():
            self.catchNextExceptionBtn.setChecked(False)
        elif not self.catchAllExceptionsBtn.isChecked():
            return
        
        self.setException(exc, lastFrame=lastFrame)
    
    def _threadExceptionHandler(self, args):
        self.exceptionHandler(*args)

    def systrace(self, frame, event, arg):
        if event != 'exception':
            return self.systrace

        if self._inSystrace:
            # prevent recursve calling
            return self.systrace
        self._inSystrace = True
        try:
            if self.checkException(*arg):
                # note: the exception has no __traceback__ at this point!
                self.exceptionHandler(arg[1], lastFrame=frame)
        except Exception as exc:
            print("Exception in systrace:")
            traceback.print_exc()
        finally:
            self._inSystrace = False
        return self.systrace
        
    def checkException(self, excType, exc, tb):
        ## Return True if the exception is interesting; False if it should be ignored.
        
        filename = tb.tb_frame.f_code.co_filename
        function = tb.tb_frame.f_code.co_name
        
        filterStr = self.filterString
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

    def _filterTextChanged(self, value):
        self.filterString = str(value)
