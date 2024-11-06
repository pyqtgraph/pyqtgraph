import code
import queue
import sys
import traceback

from ..functions import mkBrush
from ..Qt import QtCore, QtGui, QtWidgets
from .CmdInput import CmdInput


class ReplWidget(QtWidgets.QWidget):
    sigCommandEntered = QtCore.Signal(object, object)  # self, command
    sigCommandRaisedException = QtCore.Signal(object, object)  # self, exc

    def __init__(self, globals, locals, parent=None, allowNonGuiExecution=False):
        self._lastCommandRow = None

        QtWidgets.QWidget.__init__(self, parent=parent)

        self._allowNonGuiExecution = allowNonGuiExecution
        self._thread = ReplThread(self, globals, locals, parent=self)
        self._thread.sigCommandEntered.connect(self.sigCommandEntered)
        self._thread.sigCommandRaisedException.connect(self.handleException)
        self._thread.sigCommandExecuted.connect(self.handleCommandExecuted)
        if allowNonGuiExecution:
            self._thread.start()

        self._setupUi()

        # define text styles
        isDark = self.output.palette().color(QtGui.QPalette.ColorRole.Base).value() < 128
        outputBlockFormat = QtGui.QTextBlockFormat()
        outputFirstLineBlockFormat = QtGui.QTextBlockFormat(outputBlockFormat)
        outputFirstLineBlockFormat.setTopMargin(5)
        outputCharFormat = QtGui.QTextCharFormat()
        outputCharFormat.setFont(self.output.font())
        outputCharFormat.setFontWeight(QtGui.QFont.Weight.Normal)
        cmdBlockFormat = QtGui.QTextBlockFormat()
        cmdBlockFormat.setBackground(mkBrush("#335" if isDark else "#CCF"))
        cmdCharFormat = QtGui.QTextCharFormat()
        cmdCharFormat.setFont(self.output.font())
        cmdCharFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        self.textStyles = {
            'command': (cmdCharFormat, cmdBlockFormat),
            'output': (outputCharFormat, outputBlockFormat),
            'output_first_line': (outputCharFormat, outputFirstLineBlockFormat),
        }

        self.input.ps1 = self._thread.ps1
        self.input.ps2 = self._thread.ps2

    def _setupUi(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.output = QtWidgets.QTextEdit(self)
        font = QtGui.QFont("monospace")
        self.output.setFont(font)
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)
        
        # put input box in a horizontal layout so we can easily place buttons at the end
        self.inputWidget = QtWidgets.QWidget(self)
        self.layout.addWidget(self.inputWidget)
        self.inputLayout = QtWidgets.QHBoxLayout()
        self.inputWidget.setLayout(self.inputLayout)
        self.inputLayout.setContentsMargins(0, 0, 0, 0)

        self.input = CmdInput(parent=self)
        self.input.setFont(font)
        self.inputLayout.addWidget(self.input)
        if self._allowNonGuiExecution:
            self.guiCheckbox = QtWidgets.QCheckBox("Exec in GUI", self)
            self.guiCheckbox.setChecked(True)
            self.guiCheckbox.setToolTip(
                "If your command is long-running and does not require GUI interaction,"
                " uncheck this box to run it in a separate thread."
            )
            self.inputLayout.addWidget(self.guiCheckbox)

        self.input.sigExecuteCmd.connect(self.handleCommand)
        self._thread.sigInputGenerated.connect(self.write)
        self._thread.sigMultilineChanged.connect(self._setMultiline)

    def handleCommand(self, cmd):
        self.input.setEnabled(False)
        if self._allowNonGuiExecution and not self.guiCheckbox.isChecked():
            self._thread.queueCommand(cmd)
        else:
            self._thread.runCmd(cmd)

    def handleCommandExecuted(self):
        self.input.setEnabled(True)
        self.input.setFocus()

    def handleException(self, exc):
        self.input.setEnabled(True)
        self.input.setFocus()
        self.sigCommandRaisedException.emit(self, exc)

    def write(self, strn, style='output', scrollToBottom='auto'):
        """Write a string into the console.

        If scrollToBottom is 'auto', then the console is automatically scrolled
        to fit the new text only if it was already at the bottom.
        """
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            sys.__stdout__.write(strn)
            return

        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)

        sb = self.output.verticalScrollBar()
        scroll = sb.value()
        if scrollToBottom == 'auto':
            atBottom = scroll == sb.maximum()
            scrollToBottom = atBottom

        row = cursor.blockNumber()
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
            if style != 'output':
                self._setTextStyle('output')

        if scrollToBottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scroll)

    def _setMultiline(self, enable):
        self.input.setMultiline(enable)
        if enable:
            self.input.setEnabled(True)
            self.input.setFocus()

    def _setTextStyle(self, style):
        charFormat, blockFormat = self.textStyles[style]
        cursor = self.output.textCursor()
        cursor.setBlockFormat(blockFormat)
        self.output.setCurrentCharFormat(charFormat)


class ReplThread(QtCore.QThread):
    sigCommandEntered = QtCore.Signal(object, object)  # repl, command
    sigCommandRaisedException = QtCore.Signal(object)  # exception
    sigCommandExecuted = QtCore.Signal()
    sigInputGenerated = QtCore.Signal(str, str, str)  # input, style, scrollToBottom
    sigMultilineChanged = QtCore.Signal(bool)

    def __init__(self, repl, globals_, locals_, parent=None):
        QtCore.QThread.__init__(self, parent=parent)
        self._repl = repl
        self._globals = globals_
        self._locals = locals_
        self.ps1 = ">>> "
        self.ps2 = "... "
        self._stdoutInterceptor = StdoutInterceptor(self.write)
        self._commandBuffer = []  # buffer to hold multiple lines of a single command
        self._commands = queue.Queue()

    def queueCommand(self, cmd):
        self._commands.put(cmd)

    def run(self):
        # todo handle external interruptions
        while True:
            cmd = self._commands.get()
            self.runCmd(cmd)

    def runCmd(self, cmd):
        if '\n' in cmd:
            for line in cmd.split('\n'):
                self.runCmd(line)
            return

        if len(self._commandBuffer) == 0:
            self.write(f"{self.ps1}{cmd}\n", 'command')
        else:
            self.write(f"{self.ps2}{cmd}\n", 'command')

        self.sigCommandEntered.emit(self._repl, cmd)
        self._commandBuffer.append(cmd)

        fullcmd = '\n'.join(self._commandBuffer)
        try:
            cmdCode = code.compile_command(fullcmd)
            self.sigMultilineChanged.emit(False)
        except Exception as e:
            # cannot continue processing this command; reset and print exception
            self._commandBuffer = []
            self.displayException()
            self.sigMultilineChanged.emit(False)
            self.sigCommandRaisedException.emit(e)
        else:
            if cmdCode is None:
                # incomplete input; wait for next line
                self.sigMultilineChanged.emit(True)
                return

            self._commandBuffer = []

            # run command
            try:
                with self._stdoutInterceptor:
                    exec(cmdCode, self._globals(), self._locals())
                    self.sigCommandExecuted.emit()
            except Exception as exc:
                self.displayException()
                self.sigCommandRaisedException.emit(exc)

            # Add a newline if the output did not
            cursor = self._repl.output.textCursor()
            if cursor.columnNumber() > 0:
                self.write('\n')

    def write(self, strn, style='output', scrollToBottom='auto'):
        self.sigInputGenerated.emit(strn, style, scrollToBottom)

    def displayException(self):
        """Display the current exception and stack."""
        tb = traceback.format_exc()
        indent = 4
        lines = [f"{' ' * indent}{line}" for line in tb.split('\n')]
        self.write('\n'.join(lines))


class StdoutInterceptor:
    """Used to temporarily redirect writes meant for sys.stdout and sys.stderr to a new location
    """
    def __init__(self, writeFn):
        self._orig_stdout = None
        self._orig_stderr = None
        self.writeFn = writeFn

    def realOutputFiles(self):
        """Return the real sys.stdout and stderr (which are sometimes masked while running commands)
        """
        return (
            self._orig_stdout or sys.stdout,
            self._orig_stderr or sys.stderr
        )

    def print(self, *args):
        """Print to real stdout (for debugging)
        """
        self.realOutputFiles()[0].write(' '.join(map(str, args)) + "\n")

    def flush(self):
        # Need to implement this since we temporarily occlude sys.stdout
        pass

    def fileno(self):
        # Need to implement this since we temporarily occlude sys.stdout, and someone may be looking for it (faulthandler, for example)
        return 1

    def write(self, strn):
        self.writeFn(strn)

    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._orig_stdout = None
        self._orig_stderr = None
