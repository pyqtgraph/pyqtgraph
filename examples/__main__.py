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
    ('GraphicsItems', OrderedDict([
        #('PlotItem', 'PlotItem.py'),
        ('ImageItem - video', 'ImageItem.py'),
        ('ImageItem - draw', 'Draw.py'),
        ('Region-of-Interest', 'ROItypes.py'),
        ('GraphicsLayout', 'GraphicsLayout.py'),
        ('Scatter Plot', 'ScatterPlot.py'),
        ('ViewBox', 'ViewBox.py'),
        ('Arrow', 'Arrow.py'),
    ])),
    ('Widgets', OrderedDict([
        ('PlotWidget', 'PlotWidget.py'),
        ('SpinBox', '../widgets/SpinBox.py'),
        ('TreeWidget', '../widgets/TreeWidget.py'),
        ('DataTreeWidget', '../widgets/DataTreeWidget.py'),
        ('GradientWidget', '../widgets/GradientWidget.py'),
        ('TableWidget', '../widgets/TableWidget.py'),
        ('ColorButton', '../widgets/ColorButton.py'),
        ('CheckTable', '../widgets/CheckTable.py'),
        ('VerticalLabel', '../widgets/VerticalLabel.py'),
        ('JoystickButton', '../widgets/JoystickButton.py'),
    ])),
    ('ImageView', 'ImageView.py'),
    ('GraphicsScene', 'GraphicsScene.py'),
    ('Flowcharts', 'Flowchart.py'),
    ('ParameterTree', '../parametertree'),
    ('Canvas', '../canvas'),
    ('MultiPlotWidget', 'MultiPlotWidget.py'),
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
        for key, val in examples.iteritems():
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
        os.spawnl(os.P_NOWAIT, sys.executable, sys.executable, fn)
        
            
    def showFile(self):
        fn = self.currentFile()
        if fn is None:
            self.ui.codeView.clear()
            return
        text = open(fn).read()
        self.ui.codeView.setPlainText(text)

def run():
    app = QtGui.QApplication([])
    loader = ExampleLoader()
    
    if sys.flags.interactive != 1:
        app.exec_()

if __name__ == '__main__':
    run()
