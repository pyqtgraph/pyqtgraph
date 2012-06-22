
from pyqtgraph.Qt import QtCore, QtGui
import sys, re, os, time, traceback, subprocess
import pyqtgraph as pg
import template
import pyqtgraph.exceptionHandling as exceptionHandling
import pickle

class ConsoleWidget(QtGui.QWidget):
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
    - ability to have multiple interactive prompts for remotely generated processes
    """
    
    def __init__(self, parent=None, namespace=None, historyFile=None, text=None, editor=None):
        """
        ============  ============================================================================
        Arguments:
        namespace     dictionary containing the initial variables present in the default namespace
        historyFile   optional file for storing command history
        text          initial text to display in the console window
        editor        optional string for invoking code editor (called when stack trace entries are 
                      double-clicked). May contain {fileName} and {lineNum} format keys. Example:: 
                      
                        editorCommand --loadfile {fileName} --gotoline {lineNum}
        ============  =============================================================================
        """
        QtGui.QWidget.__init__(self, parent)
        if namespace is None:
            namespace = {}
        self.localNamespace = namespace
        self.editor = editor
        self.multiline = None
        self.inCmd = False
        
        self.ui = template.Ui_Form()
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
        
        self.ui.catchAllExceptionsBtn.toggled.connect(self.catchAllToggled)
        self.ui.catchNextExceptionBtn.toggled.connect(self.catchNextToggled)
        self.ui.clearExceptionBtn.clicked.connect(self.clearExceptionClicked)
        self.ui.exceptionStackList.itemClicked.connect(self.stackItemClicked)
        self.ui.exceptionStackList.itemDoubleClicked.connect(self.stackItemDblClicked)
        
        self.exceptionHandlerRunning = False
        self.currentTraceback = None
        
    def loadHistory(self):
        """Return the list of previously-invoked command strings (or None)."""
        if self.historyFile is not None:
            return pickle.load(open(self.historyFile, 'rb'))
        
    def saveHistory(self, history):
        """Store the list of previously-invoked command strings."""
        if self.historyFile is not None:
            pickle.dump(open(self.historyFile, 'wb'), history)
        
    def runCmd(self, cmd):
        #cmd = str(self.input.lastCmd)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        encCmd = re.sub(r'>', '&gt;', re.sub(r'<', '&lt;', cmd))
        encCmd = re.sub(r' ', '&nbsp;', encCmd)
        
        self.ui.historyList.addItem(cmd)
        self.saveHistory(self.input.history[1:100])
        
        try:
            sys.stdout = self
            sys.stderr = self
            if self.multiline is not None:
                self.write("<br><b>%s</b>\n"%encCmd, html=True)
                self.execMulti(cmd)
            else:
                self.write("<br><div style='background-color: #CCF'><b>%s</b>\n"%encCmd, html=True)
                self.inCmd = True
                self.execSingle(cmd)
            
            if not self.inCmd:
                self.write("</div>\n", html=True)
                
        finally:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            
            sb = self.output.verticalScrollBar()
            sb.setValue(sb.maximum())
            sb = self.ui.historyList.verticalScrollBar()
            sb.setValue(sb.maximum())
            
    def globals(self):
        frame = self.currentFrame()
        if frame is not None and self.ui.runSelectedFrameCheck.isChecked():
            return self.currentFrame().tb_frame.f_globals
        else:
            return globals()
        
    def locals(self):
        frame = self.currentFrame()
        if frame is not None and self.ui.runSelectedFrameCheck.isChecked():
            return self.currentFrame().tb_frame.f_locals
        else:
            return self.localNamespace
            
    def currentFrame(self):
        ## Return the currently selected exception stack frame (or None if there is no exception)
        if self.currentTraceback is None:
            return None
        index = self.ui.exceptionStackList.currentRow()
        tb = self.currentTraceback
        for i in range(index):
            tb = tb.tb_next
        return tb
        
    def execSingle(self, cmd):
        try:
            output = eval(cmd, self.globals(), self.locals())
            self.write(repr(output) + '\n')
        except SyntaxError:
            try:
                exec(cmd, self.globals(), self.locals())
            except SyntaxError as exc:
                if 'unexpected EOF' in exc.msg:
                    self.multiline = cmd
                else:
                    self.displayException()
            except:
                self.displayException()
        except:
            self.displayException()
            
            
    def execMulti(self, nextLine):
        self.stdout.write(nextLine+"\n")
        if nextLine.strip() != '':
            self.multiline += "\n" + nextLine
            return
        else:
            cmd = self.multiline
            
        try:
            output = eval(cmd, self.globals(), self.locals())
            self.write(str(output) + '\n')
            self.multiline = None
        except SyntaxError:
            try:
                exec(cmd, self.globals(), self.locals())
                self.multiline = None
            except SyntaxError as exc:
                if 'unexpected EOF' in exc.msg:
                    self.multiline = cmd
                else:
                    self.displayException()
                    self.multiline = None
            except:
                self.displayException()
                self.multiline = None
        except:
            self.displayException()
            self.multiline = None

    def write(self, strn, html=False):
        self.output.moveCursor(QtGui.QTextCursor.End)
        if html:
            self.output.textCursor().insertHtml(strn)
        else:
            if self.inCmd:
                self.inCmd = False
                self.output.textCursor().insertHtml("</div><br><div style='font-weight: normal; background-color: #FFF;'>")
                #self.stdout.write("</div><br><div style='font-weight: normal; background-color: #FFF;'>")
            self.output.insertPlainText(strn)
        #self.stdout.write(strn)
            
    def displayException(self):
        tb = traceback.format_exc()
        lines = []
        indent = 4
        prefix = '' 
        for l in tb.split('\n'):
            lines.append(" "*indent + prefix + l)
        self.write('\n'.join(lines))
        
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

    def catchAllToggled(self, b):
        if b:
            self.ui.catchNextExceptionBtn.setChecked(False)
            exceptionHandling.register(self.allExceptionsHandler)
        else:
            exceptionHandling.unregister(self.allExceptionsHandler)
        
    def catchNextToggled(self, b):
        if b:
            self.ui.catchAllExceptionsBtn.setChecked(False)
            exceptionHandling.register(self.nextExceptionHandler)
        else:
            exceptionHandling.unregister(self.nextExceptionHandler)
        
        
    def clearExceptionClicked(self):
        self.currentTraceback = None
        self.ui.exceptionInfoLabel.setText("[No current exception]")
        self.ui.exceptionStackList.clear()
        self.ui.clearExceptionBtn.setEnabled(False)
        
    def stackItemClicked(self, item):
        pass
    
    def stackItemDblClicked(self, item):
        global EDITOR
        tb = self.currentFrame()
        lineNum = tb.tb_lineno
        fileName = tb.tb_frame.f_code.co_filename
        subprocess.Popen(EDITOR.format(fileName=fileName, lineNum=lineNum), shell=True)
        
    
    def allExceptionsHandler(self, *args):
        self.exceptionHandler(*args)
    
    def nextExceptionHandler(self, *args):
        self.ui.catchNextExceptionBtn.setChecked(False)
        self.exceptionHandler(*args)

    def exceptionHandler(self, excType, exc, tb):
        self.ui.clearExceptionBtn.setEnabled(True)
        self.currentTraceback = tb
        
        excMessage = ''.join(traceback.format_exception_only(excType, exc))
        self.ui.exceptionInfoLabel.setText(excMessage)
        self.ui.exceptionStackList.clear()
        for index, line in enumerate(traceback.extract_tb(tb)):
            self.ui.exceptionStackList.addItem('File "%s", line %s, in %s()\n  %s' % line)
        
    
    def quit(self):
        if self.exceptionHandlerRunning:
            self.exitHandler = True
        try:
            exceptionHandling.unregister(self.exceptionHandler)
        except:
            pass
        
        