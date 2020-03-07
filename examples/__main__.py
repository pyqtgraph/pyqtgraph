import sys, os
if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"
import pyqtgraph as pg
import subprocess
from pyqtgraph.python2_3 import basestring
from pyqtgraph.Qt import QtGui, QT_LIB

from .utils import buildFileList, path, examples
from .syntax import PythonHighlighter


if QT_LIB == 'PySide':
    from .exampleLoaderTemplate_pyside import Ui_Form
elif QT_LIB == 'PySide2':
    from .exampleLoaderTemplate_pyside2 import Ui_Form
elif QT_LIB == 'PyQt5':
    from .exampleLoaderTemplate_pyqt5 import Ui_Form
else:
    from .exampleLoaderTemplate_pyqt import Ui_Form

class App(QtGui.QApplication):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paletteChanged.connect(self.onPaletteChange)
        self.onPaletteChange(self.palette())

    def onPaletteChange(self, palette):
        self.dark_mode = palette.base().color().name().lower() != "#ffffff"

class ExampleLoader(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_Form()
        self.cw = QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.ui.setupUi(self.cw)
        self.setWindowTitle("PyQtGraph Examples")

        self.codeBtn = QtGui.QPushButton('Run Edited Code')
        self.codeLayout = QtGui.QGridLayout()
        self.ui.codeView.setLayout(self.codeLayout)
        #self.simulate_black_mode()
        self.hl = PythonHighlighter(self.ui.codeView.document())
        self.codeLayout.addItem(QtGui.QSpacerItem(100,100,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding), 0, 0)
        self.codeLayout.addWidget(self.codeBtn, 1, 1)
        self.codeBtn.hide()

        global examples
        self.itemCache = []
        self.populateTree(self.ui.exampleTree.invisibleRootItem(), examples)
        self.ui.exampleTree.expandAll()

        self.resize(1000,500)
        self.show()
        self.ui.splitter.setSizes([250,750])
        self.ui.loadBtn.clicked.connect(self.loadFile)
        self.ui.exampleTree.currentItemChanged.connect(self.showFile)
        self.ui.exampleTree.itemDoubleClicked.connect(self.loadFile)
        self.ui.codeView.textChanged.connect(self.codeEdited)
        self.codeBtn.clicked.connect(self.runEditedCode)

    def simulate_black_mode(self):
        """
        used to simulate MacOS "black mode" on other platforms
        intended for debug only, as it manage only the QPlainTextEdit
        """
        # first, a dark background
        c = QtGui.QColor('#171717')
        p = self.ui.codeView.palette()
        p.setColor(QtGui.QPalette.Active, QtGui.QPalette.Base, c)
        p.setColor(QtGui.QPalette.Inactive, QtGui.QPalette.Base, c)
        self.ui.codeView.setPalette(p)
        # then, a light font
        f = QtGui.QTextCharFormat()
        f.setForeground(QtGui.QColor('white'))
        self.ui.codeView.setCurrentCharFormat(f)
        # finally, override application automatic detection
        app = QtGui.QApplication.instance()
        app.dark_mode = True

    def populateTree(self, root, examples):
        for key, val in examples.items():
            item = QtGui.QTreeWidgetItem([key])
            self.itemCache.append(item) # PyQt 4.9.6 no longer keeps references to these wrappers,
                                        # so we need to make an explicit reference or else the .file
                                        # attribute will disappear.
            if isinstance(val, basestring):
                item.file = val
            else:
                self.populateTree(item, val)
            root.addChild(item)

    def currentFile(self):
        item = self.ui.exampleTree.currentItem()
        if hasattr(item, 'file'):
            global path
            return os.path.join(path, item.file)
        return None

    def loadFile(self, edited=False):

        extra = []
        qtLib = str(self.ui.qtLibCombo.currentText())
        gfxSys = str(self.ui.graphicsSystemCombo.currentText())

        if qtLib != 'default':
            extra.append(qtLib.lower())
        elif gfxSys != 'default':
            extra.append(gfxSys)

        if edited:
            path = os.path.abspath(os.path.dirname(__file__))
            proc = subprocess.Popen([sys.executable, '-'] + extra, stdin=subprocess.PIPE, cwd=path)
            code = str(self.ui.codeView.toPlainText()).encode('UTF-8')
            proc.stdin.write(code)
            proc.stdin.close()
        else:
            fn = self.currentFile()
            if fn is None:
                return
            if sys.platform.startswith('win'):
                os.spawnl(os.P_NOWAIT, sys.executable, '"'+sys.executable+'"', '"' + fn + '"', *extra)
            else:
                os.spawnl(os.P_NOWAIT, sys.executable, sys.executable, fn, *extra)

    def showFile(self):
        fn = self.currentFile()
        if fn is None:
            self.ui.codeView.clear()
            return
        if os.path.isdir(fn):
            fn = os.path.join(fn, '__main__.py')
        text = open(fn).read()
        self.ui.codeView.setPlainText(text)
        self.ui.loadedFileLabel.setText(fn)
        self.codeBtn.hide()

    def codeEdited(self):
        self.codeBtn.show()

    def runEditedCode(self):
        self.loadFile(edited=True)

def run():
    app = App([])
    loader = ExampleLoader()
    app.exec_()

if __name__ == '__main__':
    run()
