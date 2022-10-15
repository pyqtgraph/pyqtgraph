import pickle
import re
import subprocess
import sys
import traceback
import code

from .. import exceptionHandling as exceptionHandling
from .. import getConfigOption
from ..functions import SignalBlock, mkBrush
from ..Qt import QtCore, QtGui, QtWidgets
from . import template_generic as ui_template


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
    _threadException = QtCore.Signal(object)
    
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
        if namespace is None:
            namespace = {}
        namespace['__console__'] = self
        self._orig_stdout = None
        self._orig_stderr = None
        self._lastCommandRow = None

        outputBlockFormat = QtGui.QTextBlockFormat()

        outputFirstLineBlockFormat = QtGui.QTextBlockFormat(outputBlockFormat)
        outputFirstLineBlockFormat.setTopMargin(5)

        outputCharFormat = QtGui.QTextCharFormat()
        outputCharFormat.setFontWeight(QtGui.QFont.Normal)

        cmdBlockFormat = QtGui.QTextBlockFormat()
        cmdBlockFormat.setBackground(mkBrush("#CCF"))

        cmdCharFormat = QtGui.QTextCharFormat()
        cmdCharFormat.setFontWeight(QtGui.QFont.Bold)

        self.textStyles = {
            'command': (cmdCharFormat, cmdBlockFormat),
            'output': (outputCharFormat, outputBlockFormat),
            'output_first_line': (outputCharFormat, outputFirstLineBlockFormat),
        }

        self.localNamespace = namespace
        self.editor = editor
        self._commandBuffer = []  # buffer to hold multiple lines of input
        
        self.ui = ui_template.Ui_Form()
        self.ui.setupUi(self)
        self.output = self.ui.output
        self.input = self.ui.input
        self.input.setFocus()
        
        if text is not None:
            self.output.setPlainText(text)

        self.historyFile = historyFile
        
        history = self.loadHistory()
        if history is not None:
            self.input.history = [""] + history
            self.ui.historyList.addItems(history[::-1])
        self.ui.historyList.hide()
        self.ui.exceptionGroup.hide()
        
        self.input.sigExecuteCmd.connect(self.runCmd)
        self.ui.historyBtn.toggled.connect(self.ui.historyList.setVisible)
        self.ui.historyList.itemClicked.connect(self.cmdSelected)
        self.ui.historyList.itemDoubleClicked.connect(self.cmdDblClicked)
        self.ui.exceptionBtn.toggled.connect(self.ui.exceptionGroup.setVisible)
        
        self.ui.catchAllExceptionsBtn.toggled.connect(self.catchAllExceptions)
        self.ui.catchNextExceptionBtn.toggled.connect(self.catchNextException)
        self.ui.clearExceptionBtn.clicked.connect(self.clearExceptionClicked)
        self.ui.exceptionStackList.itemClicked.connect(self.stackItemClicked)
        self.ui.exceptionStackList.itemDoubleClicked.connect(self.stackItemDblClicked)
        self.ui.onlyUncaughtCheck.toggled.connect(self.updateSysTrace)
        
        self.currentTraceback = None

        # send exceptions raised in non-gui threads back to the main thread by signal.
        self._threadException.connect(self._threadExceptionHandler)
        
    def loadHistory(self):
        """Return the list of previously-invoked command strings (or None)."""
        if self.historyFile is not None:
            with open(self.historyFile, 'rb') as pf:
                return pickle.load(pf)
        
    def saveHistory(self, history):
        """Store the list of previously-invoked command strings."""
        if self.historyFile is not None:
            with open(self.historyFile, 'wb') as pf:
                pickle.dump(pf, history)
        
    def runCmd(self, cmd):
        #cmd = str(self.input.lastCmd)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        encCmd = re.sub(r'>', '&gt;', re.sub(r'<', '&lt;', cmd))
        encCmd = re.sub(r' ', '&nbsp;', encCmd)
        
        self.ui.historyList.addItem(cmd)
        self.saveHistory(self.input.history[1:100])
        
        try:
            sys.stdout = self
            sys.stderr = self

            # jump to next line before printing commands
            cursor = self.output.textCursor()
            if cursor.columnNumber() > 0:
                self.output.insertPlainText('\n')

            if len(self._commandBuffer) == 0:
                # start a new div to display this command                
                # self.write(f'<div style="{self.commandStyle}">&gt;&gt;&gt; <b>{encCmd}</b></div>', html=True)
                self.write(f">>> {cmd}\n", style='command')
            else:
                # self.write(f'<span style="{self.commandStyle}">... <b>{encCmd}</b></span><br>', html=True)
                self.write(f"... {cmd}\n", style='command')
            
            self._commandBuffer.append(cmd)

            cmd = '\n'.join(self._commandBuffer)
            try:
                cmdCode = code.compile_command(cmd)
            except Exception:
                # cannot continue processing this command; reset, close the command div, and print exception
                self._commandBuffer = []
                # self.write("</div>\n", html=True, scrollToBottom=True)
                self.displayException()
            else:
                if cmdCode is None:
                    # incomplete input; wait for next line
                    return
                # reset, close the command div, and execute code
                self._commandBuffer = []
                # self.write("</div>\n", html=True, scrollToBottom=True)

                # start a new div for command output
                # self.write(f'<br><div style="{self.outputStyle}">\n', html=True)
                # run command
                try:
                    exec(cmdCode, self.globals(), self.locals())
                except:
                    self.displayException()
                # close output div
                # self.write(f'\n</div>\n', html=True)                
                
        finally:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._orig_stdout = None
            self._orig_stderr = None
            
            sb = self.ui.historyList.verticalScrollBar()
            sb.setValue(sb.maximum())
            
    def globals(self):
        frame = self.ui.exceptionStackList.selectedFrame()
        if frame is not None and self.ui.runSelectedFrameCheck.isChecked():
            return frame.f_globals
        else:
            return self.localNamespace
        
    def locals(self):
        frame = self.ui.exceptionStackList.selectedFrame()
        if frame is not None and self.ui.runSelectedFrameCheck.isChecked():
            return frame.f_locals
        else:
            return self.localNamespace

    def realOutputFiles(self):
        """Return the real sys.stdout and stderr (which are sometimes masked while running commands)
        """
        return (
            self._orig_stdout or sys.stdout,
            self._orig_stderr or sys.stderr
        )

    def _print(self, strn):
        """Print to real stdout (for debugging)
        """
        self.realOutputFiles()[0].write(strn + "\n")

    def write(self, strn, style='output', scrollToBottom='auto'):
        """Write a string into the console.

        If scrollToBottom is 'auto', then the console is automatically scrolled
        to fit the new text only if it was already at the bottom.
        """
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            sys.__stdout__.write(strn)
            return

        sb = self.output.verticalScrollBar()
        scroll = sb.value()
        if scrollToBottom == 'auto':
            atBottom = scroll == sb.maximum()
            scrollToBottom = atBottom

        self.output.moveCursor(QtGui.QTextCursor.MoveOperation.End)

        row = self.output.textCursor().blockNumber()
        if style == 'command':
            self._lastCommandRow = row

        if style == 'output' and row == self._lastCommandRow + 1:
            # adjust style for first line of output
            firstLine, endl, strn = strn.partition('\n')
            self._setTextStyle('output_first_line')
            self.output.insertPlainText(firstLine + endl)

        if len(strn) > 0:
            self._setTextStyle(style)
            self.output.insertPlainText(strn)
            # return to output style immediately to avoid seeing an extra line of command style
            if 'style' != 'output':
                self._setTextStyle('output')

        if scrollToBottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scroll)

    def _setTextStyle(self, style):
        charFormat, blockFormat = self.textStyles[style]
        cursor = self.output.textCursor()
        cursor.setBlockFormat(blockFormat)
        self.output.setCurrentCharFormat(charFormat)

    def fileno(self):
        # Need to implement this since we temporarily occlude sys.stdout, and someone may be looking for it (faulthandler, for example)
        return 1

    def displayException(self):
        """
        Display the current exception and stack.
        """
        tb = traceback.format_exc()
        lines = []
        indent = 4
        prefix = '' 
        for l in tb.split('\n'):
            lines.append(" "*indent + prefix + l)
        self.write('\n'.join(lines))
        self.exceptionHandler(*sys.exc_info())
        
    def cmdSelected(self, item):
        index = -(self.ui.historyList.row(item)+1)
        self.input.setHistory(index)
        self.input.setFocus()
        
    def cmdDblClicked(self, item):
        index = -(self.ui.historyList.row(item)+1)
        self.input.setHistory(index)
        self.input.execCmd()
        
    def flush(self):
        pass

    def catchAllExceptions(self, catch=True):
        """
        If True, the console will catch all unhandled exceptions and display the stack
        trace. Each exception caught clears the last.
        """
        with SignalBlock(self.ui.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
            self.ui.catchAllExceptionsBtn.setChecked(catch)
        
        if catch:
            with SignalBlock(self.ui.catchNextExceptionBtn.toggled, self.catchNextException):
                self.ui.catchNextExceptionBtn.setChecked(False)
            self.enableExceptionHandling()
            self.ui.exceptionBtn.setChecked(True)
        else:
            self.disableExceptionHandling()
        
    def catchNextException(self, catch=True):
        """
        If True, the console will catch the next unhandled exception and display the stack
        trace.
        """
        with SignalBlock(self.ui.catchNextExceptionBtn.toggled, self.catchNextException):
            self.ui.catchNextExceptionBtn.setChecked(catch)
        if catch:
            with SignalBlock(self.ui.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
                self.ui.catchAllExceptionsBtn.setChecked(False)
            self.enableExceptionHandling()
            self.ui.exceptionBtn.setChecked(True)
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
        self.ui.exceptionInfoLabel.setText("[No current exception]")
        self.ui.exceptionStackList.clear()
        self.ui.clearExceptionBtn.setEnabled(False)
        
    def stackItemClicked(self, item):
        pass
    
    def stackItemDblClicked(self, item):
        editor = self.editor
        if editor is None:
            editor = getConfigOption('editorCommand')
        if editor is None:
            return
        tb = self.ui.exceptionStackList.selectedFrame()
        lineNum = tb.f_lineno
        fileName = tb.f_code.co_filename
        subprocess.Popen(self.editor.format(fileName=fileName, lineNum=lineNum), shell=True)
        
    def updateSysTrace(self):
        ## Install or uninstall  sys.settrace handler 
        
        if not self.ui.catchNextExceptionBtn.isChecked() and not self.ui.catchAllExceptionsBtn.isChecked():
            if sys.gettrace() == self.systrace:
                sys.settrace(None)
            return
        
        if self.ui.onlyUncaughtCheck.isChecked():
            if sys.gettrace() == self.systrace:
                sys.settrace(None)
        else:
            if sys.gettrace() is not None and sys.gettrace() != self.systrace:
                self.ui.onlyUncaughtCheck.setChecked(False)
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

        if self.ui.catchNextExceptionBtn.isChecked():
            self.ui.catchNextExceptionBtn.setChecked(False)
        elif not self.ui.catchAllExceptionsBtn.isChecked():
            return
        
        self.currentTraceback = tb
        
        excMessage = ''.join(traceback.format_exception_only(excType, exc))
        self.ui.exceptionInfoLabel.setText(excMessage)

        if systrace:
            # exceptions caught using systrace don't need the usual 
            # call stack + traceback handling
            self.setStack(frame.f_back.f_back)
        else:
            self.setStack(frame=frame.f_back, tb=tb)
    
    def _threadExceptionHandler(self, args):
        self.exceptionHandler(*args)

    def setStack(self, frame=None, tb=None):
        self.ui.clearExceptionBtn.setEnabled(True)
        self.ui.exceptionStackList.setStack(frame, tb)

    def systrace(self, frame, event, arg):
        if event == 'exception' and self.checkException(*arg):
            self.exceptionHandler(*arg, systrace=True)
        return self.systrace
        
    def checkException(self, excType, exc, tb):
        ## Return True if the exception is interesting; False if it should be ignored.
        
        filename = tb.tb_frame.f_code.co_filename
        function = tb.tb_frame.f_code.co_name
        
        filterStr = str(self.ui.filterText.text())
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
        if excType is KeyError:
            if filename.endswith('python2.7/weakref.py') and function in ('__contains__', 'get'):
                return False
            if filename.endswith('python2.7/copy.py') and function == '_keep_alive':
                return False
        if excType is AttributeError:
            if filename.endswith('python2.7/collections.py') and function == '__init__':
                return False
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
            if filename.endswith('pyqtgraph/functions.py') and function == 'makeQImage':
                return False
        if excType is TypeError:
            if filename.endswith('numpy/lib/function_base.py') and function == 'iterable':
                return False
        if excType is ZeroDivisionError:
            if filename.endswith('python2.7/traceback.py'):
                return False
            
        return True
    
