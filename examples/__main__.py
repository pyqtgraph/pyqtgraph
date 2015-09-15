import sys, os
if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"
import pyqtgraph as pg
import subprocess
from pyqtgraph.python2_3 import basestring
from pyqtgraph.Qt import QtGui, USE_PYSIDE, USE_PYQT5


from .utils import buildFileList, testFile, path, examples

if USE_PYSIDE:
    from .exampleLoaderTemplate_pyside import Ui_Form
elif USE_PYQT5:
    from .exampleLoaderTemplate_pyqt5 import Ui_Form
else:
    from .exampleLoaderTemplate_pyqt import Ui_Form

class ExampleLoader(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_Form()
        self.cw = QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.ui.setupUi(self.cw)

        self.codeBtn = QtGui.QPushButton('Run Edited Code')
        self.codeLayout = QtGui.QGridLayout()
        self.ui.codeView.setLayout(self.codeLayout)
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
    app = QtGui.QApplication([])
    loader = ExampleLoader()

    app.exec_()

if __name__ == '__main__':

    args = sys.argv[1:]

    if '--test' in args:
        # get rid of orphaned cache files first
        pg.renamePyc(path)

        files = buildFileList(examples)
        if '--pyside' in args:
            lib = 'PySide'
        elif '--pyqt' in args or '--pyqt4' in args:
            lib = 'PyQt4'
        elif '--pyqt5' in args:
            lib = 'PyQt5'
        else:
            lib = ''

        exe = sys.executable
        print("Running tests:", lib, sys.executable)
        for f in files:
            testFile(f[0], f[1], exe, lib)
    else:
        run()
