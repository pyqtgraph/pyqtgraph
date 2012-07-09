import sys, os
## make sure this pyqtgraph is importable before any others
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyqtgraph.Qt import QtCore, QtGui

from exampleLoaderTemplate import Ui_Form
import os, sys
from collections import OrderedDict

examples = OrderedDict([
    ('Command-line usage', 'CLIexample.py'),
    ('Basic Plotting', 'Plotting.py'),
    ('ImageView', 'ImageView.py'),
    ('ParameterTree', '../parametertree'),
    ('Crosshair / Mouse interaction', 'crosshair.py'),
    ('Video speed test', 'VideoSpeedTest.py'),
    ('Plot speed test', 'PlotSpeedTest.py'),
    ('Data Slicing', 'DataSlicing.py'),
    ('Plot Customization', 'customPlot.py'),
    ('GraphicsItems', OrderedDict([
        ('Scatter Plot', 'ScatterPlot.py'),
        #('PlotItem', 'PlotItem.py'),
        ('IsocurveItem', 'isocurve.py'),
        ('ImageItem - video', 'ImageItem.py'),
        ('ImageItem - draw', 'Draw.py'),
        ('Region-of-Interest', 'ROIExamples.py'),
        ('GraphicsLayout', 'GraphicsLayout.py'),
        ('Text Item', 'text.py'),
        ('Linked Views', 'linkedViews.py'),
        ('Arrow', 'Arrow.py'),
        ('ViewBox', 'ViewBox.py'),
    ])),
    ('3D Graphics', OrderedDict([
        ('Volumetric', 'GLVolumeItem.py'),
        ('Isosurface', 'GLMeshItem.py'),
    ])),
    ('Widgets', OrderedDict([
        ('PlotWidget', 'PlotWidget.py'),
        ('SpinBox', 'SpinBox.py'),
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
        
        self.resize(900,500)
        self.show()
        self.ui.splitter.setSizes([150,750])
        self.ui.loadBtn.clicked.connect(self.loadFile)
        self.ui.exampleTree.currentItemChanged.connect(self.showFile)
        self.ui.exampleTree.itemDoubleClicked.connect(self.loadFile)


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
        if fn is None:
            return
        if sys.platform.startswith('win'):
            os.spawnl(os.P_NOWAIT, sys.executable, sys.executable, '"' + fn + '"')
        else:
            os.spawnl(os.P_NOWAIT, sys.executable, sys.executable, fn)
        
            
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

if __name__ == '__main__':
    run()
