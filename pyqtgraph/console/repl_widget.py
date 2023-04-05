import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput


class ReplWidget(QtWidgets.QWidget):
    sigCommandEntered = QtCore.Signal(object, object)  # self, command
    sigCommandRaisedException = QtCore.Signal(object, object)  # self, exc

    def __init__(self, globals, locals, parent=None):
        self.globals = globals
        self.locals = locals
        self._lastCommandRow = None
        self._commandBuffer = []  # buffer to hold multiple lines of input
        self.stdoutInterceptor = StdoutInterceptor(self.write)
        self.ps1 = ">>> "
        self.ps2 = "... "

        QtWidgets.QWidget.__init__(self, parent=parent)

        self._setupUi()

        # define text styles
        isDark = self.output.palette().color(QtGui.QPalette.ColorRole.Base).value() < 128
        outputBlockFormat = QtGui.QTextBlockFormat()
        outputFirstLineBlockFormat = QtGui.QTextBlockFormat(outputBlockFormat)
        outputFirstLineBlockFormat.setTopMargin(5)
        outputCharFormat = QtGui.QTextCharFormat()
        outputCharFormat.setFontWeight(QtGui.QFont.Weight.Normal)
        cmdBlockFormat = QtGui.QTextBlockFormat()
        cmdBlockFormat.setBackground(mkBrush("#335" if isDark else "#CCF"))
        cmdCharFormat = QtGui.QTextCharFormat()
        cmdCharFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        self.textStyles = {
            'command': (cmdCharFormat, cmdBlockFormat),
            'output': (outputCharFormat, outputBlockFormat),
            'output_first_line': (outputCharFormat, outputFirstLineBlockFormat),
        }

        self.input.ps1 = self.ps1
        self.input.ps2 = self.ps2

    def _setupUi(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.output = QtWidgets.QTextEdit(self)
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
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
        self.inputLayout.addWidget(self.input)

        self.input.sigExecuteCmd.connect(self.runCmd)

    def runCmd(self, cmd):
        if '\n' in cmd:
            for line in cmd.split('\n'):
                self.runCmd(line)
            return

        if len(self._commandBuffer) == 0:
            self.write(f"{self.ps1}{cmd}\n", style='command')
        else:
            self.write(f"{self.ps2}{cmd}\n", style='command')
        
        self.sigCommandEntered.emit(self, cmd)
        self._commandBuffer.append(cmd)

        fullcmd = '\n'.join(self._commandBuffer)
        try:
            cmdCode = code.compile_command(fullcmd)
            self.input.setMultiline(False)
        except Exception:
            # cannot continue processing this command; reset and print exception
            self._commandBuffer = []
            self.displayException()
            self.input.setMultiline(False)
        else:
            if cmdCode is None:
                # incomplete input; wait for next line
                self.input.setMultiline(True)
                return

            self._commandBuffer = []

            # run command
            try:
                with self.stdoutInterceptor:
                    exec(cmdCode, self.globals(), self.locals())
            except Exception as exc:
                self.displayException()
                self.sigCommandRaisedException.emit(self, exc)

            # Add a newline if the output did not
            cursor = self.output.textCursor()
            if cursor.columnNumber() > 0:
                self.write('\n')

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
        self.output.setTextCursor(cursor)

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
        
    def _setTextStyle(self, style):
        charFormat, blockFormat = self.textStyles[style]
        cursor = self.output.textCursor()
        cursor.setBlockFormat(blockFormat)
        self.output.setCurrentCharFormat(charFormat)


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

