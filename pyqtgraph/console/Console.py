import os
import sys
import pickle
import subprocess

from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget


class ConsoleWidget(QtWidgets.QWidget):
    """
    Widget displaying console output and accepting command input.
    Implements:
        
      - eval python expressions / exec python statements
      - storable history of commands
      - exception handling allowing commands to be interpreted in the context of any level in the exception stack frame
    
    Why not just use python in an interactive shell (or ipython) ? There are a few reasons:
       
      - pyside does not yet allow Qt event processing and interactive shell at the same time
      - on some systems, typing in the console _blocks_ the qt event loop until the user presses enter. This can
        be baffling and frustrating to users since it would appear the program has frozen.
      - some terminals (eg windows cmd.exe) have notoriously unfriendly interfaces
      - ability to add extra features like exception stack introspection
      - ability to have multiple interactive prompts, including for spawned sub-processes
    """
    def __init__(self, parent=None, namespace=None, historyFile=None, text=None, editor=None):
        """
        ==============  =============================================================================
        **Arguments:**
        namespace       dictionary containing the initial variables present in the default namespace
        historyFile     optional file for storing command history
        text            initial text to display in the console window
        editor          optional string for invoking code editor (called when stack trace entries are 
                        double-clicked). May contain {fileName} and {lineNum} format keys. Example:: 
                      
                            editorCommand --loadfile {fileName} --gotoline {lineNum}
        ==============  =============================================================================
        """
        QtWidgets.QWidget.__init__(self, parent)

        self._setupUi()

        if namespace is None:
            namespace = {}
        namespace['__console__'] = self

        self.localNamespace = namespace
        self.editor = editor
        
        self.output = self.repl.output
        self.input = self.repl.input
        self.input.setFocus()
        
        if text is not None:
            self.output.setPlainText(text)

        self.historyFile = historyFile
        
        try:
            history = self.loadHistory()
        except Exception as exc:
            sys.excepthook(*sys.exc_info())
            history = None
        if history is not None:
            self.input.history = [""] + history
            self.historyList.addItems(history[::-1])

        self.currentTraceback = None

    def _setupUi(self):
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        self.layout.addWidget(self.splitter, 0, 0)

        self.repl = ReplWidget(self.globals, self.locals, self)
        self.splitter.addWidget(self.repl)

        self.historyList = QtWidgets.QListWidget(self)
        self.historyList.hide()
        self.splitter.addWidget(self.historyList)

        self.historyBtn = QtWidgets.QPushButton('History', self)
        self.historyBtn.setCheckable(True)
        self.repl.inputLayout.addWidget(self.historyBtn)

        self.repl.sigCommandEntered.connect(self._commandEntered)
        self.repl.sigCommandRaisedException.connect(self._commandRaisedException)

        self.excHandler = ExceptionHandlerWidget(self)
        self.excHandler.hide()
        self.splitter.addWidget(self.excHandler)

        self.exceptionBtn = QtWidgets.QPushButton("Exceptions..", self)
        self.exceptionBtn.setCheckable(True)
        self.repl.inputLayout.addWidget(self.exceptionBtn)

        self.excHandler.sigStackItemDblClicked.connect(self._stackItemDblClicked)
        self.exceptionBtn.toggled.connect(self.excHandler.setVisible)
        self.historyBtn.toggled.connect(self.historyList.setVisible)
        self.historyList.itemClicked.connect(self.cmdSelected)
        self.historyList.itemDoubleClicked.connect(self.cmdDblClicked)

    def catchAllExceptions(self, catch=True):
        if catch:
            self.exceptionBtn.setChecked(True)
        self.excHandler.catchAllExceptions(catch)

    def catchNextException(self, catch=True):
        if catch:
            self.exceptionBtn.setChecked(True)
        self.excHandler.catchNextException(catch)

    def setStack(self, frame=None):
        self.excHandler.setStack(frame)
 
    def loadHistory(self):
        """Return the list of previously-invoked command strings (or None)."""
        if self.historyFile is not None and os.path.exists(self.historyFile):
            with open(self.historyFile, 'rb') as pf:
                return pickle.load(pf)
        
    def saveHistory(self, history):
        """Store the list of previously-invoked command strings."""
        if self.historyFile is not None:
            with open(self.historyFile, 'wb') as pf:
                pickle.dump(history, pf)
        
    def _commandEntered(self, repl, cmd):
        self.historyList.addItem(cmd)
        self.saveHistory(self.input.history[1:100])
        sb = self.historyList.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _commandRaisedException(self, repl, exc):
        self.excHandler.exceptionHandler(exc)

    def globals(self):
        frame = self.excHandler.selectedFrame()
        if frame is not None and self.excHandler.runSelectedFrameCheck.isChecked():
            return frame.f_globals
        else:
            return self.localNamespace
        
    def locals(self):
        frame = self.excHandler.selectedFrame()
        if frame is not None and self.excHandler.runSelectedFrameCheck.isChecked():
            return frame.f_locals
        else:
            return self.localNamespace

    def cmdSelected(self, item):
        index = -(self.historyList.row(item)+1)
        self.input.setHistory(index)
        self.input.setFocus()
        
    def cmdDblClicked(self, item):
        index = -(self.historyList.row(item)+1)
        self.input.setHistory(index)
        self.input.execCmd()
        
    def _stackItemDblClicked(self, handler, item):
        editor = self.editor
        if editor is None:
            editor = getConfigOption('editorCommand')
        if editor is None:
            return
        tb = self.excHandler.selectedFrame()
        lineNum = tb.f_lineno
        fileName = tb.f_code.co_filename
        subprocess.Popen(self.editor.format(fileName=fileName, lineNum=lineNum), shell=True)
