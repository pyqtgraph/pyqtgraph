import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput


class ReplWidget(QtWidgets.QWidget):
    sigCommandEntered = QtCore.Signal(object, object)  # self, command
    sigCommandRaisedException = QtCore.Signal(object, object)  # self, exc_info

    def __init__(self, globals, locals, parent=None):
        self.globals = globals
        self.locals = locals
        self._orig_stdout = None
        self._orig_stderr = None
        self._lastCommandRow = None
        self._commandBuffer = []  # buffer to hold multiple lines of input

        QtWidgets.QWidget.__init__(self, parent=parent)

        self._setupUi()

        # define text styles
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

    def _setupUi(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.output = QtWidgets.QTextEdit(self)
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
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
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        
        try:
            sys.stdout = self
            sys.stderr = self

            # jump to next line before printing commands
            cursor = self.output.textCursor()
            if cursor.columnNumber() > 0:
                self.output.insertPlainText('\n')

            if len(self._commandBuffer) == 0:
                self.write(f">>> {cmd}\n", style='command')
            else:
                self.write(f"... {cmd}\n", style='command')
            
            self._commandBuffer.append(cmd)

            fullcmd = '\n'.join(self._commandBuffer)
            try:
                cmdCode = code.compile_command(fullcmd)
            except Exception:
                # cannot continue processing this command; reset and print exception
                self._commandBuffer = []
                self.displayException()
            else:
                if cmdCode is None:
                    # incomplete input; wait for next line
                    return

                self._commandBuffer = []

                # run command
                try:
                    exec(cmdCode, self.globals(), self.locals())
                except:
                    self.displayException()
                    self.sigCommandRaisedException.emit(self, sys.exc_info())
                
        finally:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._orig_stdout = None
            self._orig_stderr = None
            self.sigCommandEntered.emit(self, cmd)
    
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

    def flush(self):
        # Need to implement this since we temporarily occlude sys.stdout
        pass

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
        
    def _setTextStyle(self, style):
        charFormat, blockFormat = self.textStyles[style]
        cursor = self.output.textCursor()
        cursor.setBlockFormat(blockFormat)
        self.output.setCurrentCharFormat(charFormat)

