import sys, os, subprocess, time
## make sure this pyqtgraph is importable before any others
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyqtgraph.Qt import QtCore, QtGui, USE_PYSIDE

if USE_PYSIDE:
    from exampleLoaderTemplate_pyside import Ui_Form
else:
    from exampleLoaderTemplate_pyqt import Ui_Form
    
import os, sys
from pyqtgraph.pgcollections import OrderedDict

examples = OrderedDict([
    ('Command-line usage', 'CLIexample.py'),
    ('Basic Plotting', 'Plotting.py'),
    ('ImageView', 'ImageView.py'),
    ('ParameterTree', 'parametertree.py'),
    ('Crosshair / Mouse interaction', 'crosshair.py'),
    ('Data Slicing', 'DataSlicing.py'),
    ('Plot Customization', 'customPlot.py'),
    ('Dock widgets', 'dockarea.py'),
    ('Console', 'ConsoleWidget.py'),
    ('Histograms', 'histogram.py'),
    ('GraphicsItems', OrderedDict([
        ('Scatter Plot', 'ScatterPlot.py'),
        #('PlotItem', 'PlotItem.py'),
        ('IsocurveItem', 'isocurve.py'),
        ('ImageItem - video', 'ImageItem.py'),
        ('ImageItem - draw', 'Draw.py'),
        ('Region-of-Interest', 'ROIExamples.py'),
        ('GraphicsLayout', 'GraphicsLayout.py'),
        ('LegendItem', 'Legend.py'),
        ('Text Item', 'text.py'),
        ('Linked Views', 'linkedViews.py'),
        ('Arrow', 'Arrow.py'),
        ('ViewBox', 'ViewBox.py'),
    ])),
    ('Benchmarks', OrderedDict([
        ('Video speed test', 'VideoSpeedTest.py'),
        ('Line Plot update', 'PlotSpeedTest.py'),
        ('Scatter Plot update', 'ScatterPlotSpeedTest.py'),
    ])),
    ('3D Graphics', OrderedDict([
        ('Volumetric', 'GLVolumeItem.py'),
        ('Isosurface', 'GLIsosurface.py'),
        ('Surface Plot', 'GLSurfacePlot.py'),
        ('Scatter Plot', 'GLScatterPlotItem.py'),
        ('Shaders', 'GLshaders.py'),
        ('Mesh', 'GLMeshItem.py'),
        ('Image', 'GLImageItem.py'),
    ])),
    ('Widgets', OrderedDict([
        ('PlotWidget', 'PlotWidget.py'),
        ('SpinBox', 'SpinBox.py'),
        ('ConsoleWidget', 'ConsoleWidget.py'),
        ('TreeWidget', 'TreeWidget.py'),
        ('DataTreeWidget', 'DataTreeWidget.py'),
        ('GradientWidget', 'GradientWidget.py'),
        #('TableWidget', '../widgets/TableWidget.py'),
        ('ColorButton', 'ColorButton.py'),
        #('CheckTable', '../widgets/CheckTable.py'),
        #('VerticalLabel', '../widgets/VerticalLabel.py'),
        ('JoystickButton', 'JoystickButton.py'),
    ])),
    
    ('GraphicsScene', 'GraphicsScene.py'),
    ('Flowcharts', 'Flowchart.py'),
    #('Canvas', '../canvas'),
    #('MultiPlotWidget', 'MultiPlotWidget.py'),
])

path = os.path.abspath(os.path.dirname(__file__))

class ExampleLoader(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_Form()
        self.cw = QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.ui.setupUi(self.cw)
        
        global examples
        self.populateTree(self.ui.exampleTree.invisibleRootItem(), examples)
        self.ui.exampleTree.expandAll()
        
        self.resize(1000,500)
        self.show()
        self.ui.splitter.setSizes([250,750])
        self.ui.loadBtn.clicked.connect(self.loadFile)
        self.ui.exampleTree.currentItemChanged.connect(self.showFile)
        self.ui.exampleTree.itemDoubleClicked.connect(self.loadFile)
        self.ui.pyqtCheck.toggled.connect(self.pyqtToggled)
        self.ui.pysideCheck.toggled.connect(self.pysideToggled)

    def pyqtToggled(self, b):
        if b:
            self.ui.pysideCheck.setChecked(False)
        
    def pysideToggled(self, b):
        if b:
            self.ui.pyqtCheck.setChecked(False)
        

    def populateTree(self, root, examples):
        for key, val in examples.items():
            item = QtGui.QTreeWidgetItem([key])
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
    
    def loadFile(self):
        fn = self.currentFile()
        extra = []
        if self.ui.pyqtCheck.isChecked():
            extra.append('pyqt')
        elif self.ui.pysideCheck.isChecked():
            extra.append('pyside')

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

def run():
    app = QtGui.QApplication([])
    loader = ExampleLoader()
    
    app.exec_()

def buildFileList(examples, files=None):
    if files == None:
        files = []
    for key, val in examples.items():
        #item = QtGui.QTreeWidgetItem([key])
        if isinstance(val, basestring):
            #item.file = val
            files.append((key,val))
        else:
            buildFileList(val, files)
    return files
            
def testFile(name, f, exe, lib):
    global path
    fn =  os.path.join(path,f)
    #print "starting process: ", fn
    
    sys.stdout.write(name)
    sys.stdout.flush()
    
    code = """
try:
    %s
    import %s
    print("test complete")
    import pyqtgraph as pg
    while True:  ## run a little event loop
        pg.QtGui.QApplication.processEvents()
        time.sleep(0.01)
except:
    print("test failed")
    raise

"""  % ("import %s" % lib if lib != '' else "", os.path.splitext(os.path.split(fn)[1])[0])
    #print code
    process = subprocess.Popen(['%s -i' % (exe)], shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    process.stdin.write(code.encode('UTF-8'))
    #process.stdin.close()
    output = ''
    fail = False
    while True:
        c = process.stdout.read(1).decode()
        output += c
        #sys.stdout.write(c)
        #sys.stdout.flush()
        if output.endswith('test complete'):
            break
        if output.endswith('test failed'):
            fail = True
            break
    time.sleep(1)
    process.terminate()
    res = process.communicate()
    #if 'exception' in res[1].lower() or 'error' in res[1].lower():
    if fail:
        print('.' * (50-len(name)) + 'FAILED')
        print(res[0].decode())
        print(res[1].decode())
    else:
        print('.' * (50-len(name)) + 'passed')
    


if __name__ == '__main__':
    if '--test' in sys.argv[1:]:
        files = buildFileList(examples)
        if '--pyside' in sys.argv[1:]:
            lib = 'PySide'
        elif '--pyqt' in sys.argv[1:]:
            lib = 'PyQt4'
        else:
            lib = ''
            
        exe = sys.executable
        print("Running tests:", lib, sys.executable)
        for f in files:
            testFile(f[0], f[1], exe, lib)
    else: 
        run()
