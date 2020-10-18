import os
import sys
import subprocess
import pyqtgraph as pg
from pyqtgraph.python2_3 import basestring
from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
from pyqtgraph.pgcollections import OrderedDict

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)
app = pg.mkQApp()

if QT_LIB == 'PySide':
    from exampleLoaderTemplate_pyside import Ui_Form
elif QT_LIB == 'PySide2':
    from exampleLoaderTemplate_pyside2 import Ui_Form
elif QT_LIB == 'PyQt5':
    from exampleLoaderTemplate_pyqt5 import Ui_Form
else:
    from exampleLoaderTemplate_pyqt import Ui_Form

examples = OrderedDict([
    ('Command-line usage', 'CLIexample.py'),
    ('Basic Plotting', 'Plotting.py'),
    ('ImageView', 'ImageView.py'),
    ('ParameterTree', 'parametertree.py'),
    ('Crosshair / Mouse interaction', 'crosshair.py'),
    ('Data Slicing', 'DataSlicing.py'),
    ('Plot Customization', 'customPlot.py'),
    ('Timestamps on x axis', 'DateAxisItem.py'),
    ('Image Analysis', 'imageAnalysis.py'),
    ('ViewBox Features', 'ViewBoxFeatures.py'),
    ('Dock widgets', 'dockarea.py'),
    ('Console', 'ConsoleWidget.py'),
    ('Histograms', 'histogram.py'),
    ('Beeswarm plot', 'beeswarm.py'),
    ('Symbols', 'Symbols.py'),
    ('Auto-range', 'PlotAutoRange.py'),
    ('Remote Plotting', 'RemoteSpeedTest.py'),
    ('Scrolling plots', 'scrollingPlots.py'),
    ('HDF5 big data', 'hdf5.py'),
    ('Demos', OrderedDict([
        ('Optics', 'optics_demos.py'),
        ('Special relativity', 'relativity_demo.py'),
        ('Verlet chain', 'verlet_chain_demo.py'),
        ('Koch Fractal', 'fractal.py'),
    ])),
    ('GraphicsItems', OrderedDict([
        ('Scatter Plot', 'ScatterPlot.py'),
        #('PlotItem', 'PlotItem.py'),
        ('IsocurveItem', 'isocurve.py'),
        ('GraphItem', 'GraphItem.py'),
        ('ErrorBarItem', 'ErrorBarItem.py'),
        ('FillBetweenItem', 'FillBetweenItem.py'),
        ('ImageItem - video', 'ImageItem.py'),
        ('ImageItem - draw', 'Draw.py'),
        ('Region-of-Interest', 'ROIExamples.py'),
        ('Bar Graph', 'BarGraphItem.py'),
        ('GraphicsLayout', 'GraphicsLayout.py'),
        ('LegendItem', 'Legend.py'),
        ('Text Item', 'text.py'),
        ('Linked Views', 'linkedViews.py'),
        ('Arrow', 'Arrow.py'),
        ('ViewBox', 'ViewBoxFeatures.py'),
        ('Custom Graphics', 'customGraphicsItem.py'),
        ('Labeled Graph', 'CustomGraphItem.py'),
    ])),
    ('Benchmarks', OrderedDict([
        ('Video speed test', 'VideoSpeedTest.py'),
        ('Line Plot update', 'PlotSpeedTest.py'),
        ('Scatter Plot update', 'ScatterPlotSpeedTest.py'),
        ('Multiple plots', 'MultiPlotSpeedTest.py'),
    ])),
    ('3D Graphics', OrderedDict([
        ('Volumetric', 'GLVolumeItem.py'),
        ('Isosurface', 'GLIsosurface.py'),
        ('Surface Plot', 'GLSurfacePlot.py'),
        ('Scatter Plot', 'GLScatterPlotItem.py'),
        ('Shaders', 'GLshaders.py'),
        ('Line Plot', 'GLLinePlotItem.py'),
        ('Mesh', 'GLMeshItem.py'),
        ('Image', 'GLImageItem.py'),
    ])),
    ('Widgets', OrderedDict([
        ('PlotWidget', 'PlotWidget.py'),
        ('SpinBox', 'SpinBox.py'),
        ('ConsoleWidget', 'ConsoleWidget.py'),
        ('Histogram / lookup table', 'HistogramLUT.py'),
        ('TreeWidget', 'TreeWidget.py'),
        ('ScatterPlotWidget', 'ScatterPlotWidget.py'),
        ('DataTreeWidget', 'DataTreeWidget.py'),
        ('GradientWidget', 'GradientWidget.py'),
        ('TableWidget', 'TableWidget.py'),
        ('ColorButton', 'ColorButton.py'),
        #('CheckTable', '../widgets/CheckTable.py'),
        #('VerticalLabel', '../widgets/VerticalLabel.py'),
        ('JoystickButton', 'JoystickButton.py'),
    ])),
    ('Flowcharts', 'Flowchart.py'),
    ('Custom Flowchart Nodes', 'FlowchartCustomNode.py'),
])



# based on https://github.com/art1415926535/PyQt5-syntax-highlighting

QRegExp = QtCore.QRegExp

QFont = QtGui.QFont
QColor = QtGui.QColor
QTextCharFormat = QtGui.QTextCharFormat
QSyntaxHighlighter = QtGui.QSyntaxHighlighter


def format(color, style=''):
    """
    Return a QTextCharFormat with the given attributes.
    """
    _color = QColor()
    if type(color) is not str:
        _color.setRgb(color[0], color[1], color[2])
    else:
        _color.setNamedColor(color)

    _format = QTextCharFormat()
    _format.setForeground(_color)
    if 'bold' in style:
        _format.setFontWeight(QFont.Bold)
    if 'italic' in style:
        _format.setFontItalic(True)

    return _format


class LightThemeColors:

    Red = "#B71C1C"
    Pink = "#FCE4EC"
    Purple = "#4A148C"
    DeepPurple = "#311B92"
    Indigo = "#1A237E"
    Blue = "#0D47A1"
    LightBlue = "#01579B"
    Cyan = "#006064"
    Teal = "#004D40"
    Green = "#1B5E20"
    LightGreen = "#33691E"
    Lime = "#827717"
    Yellow = "#F57F17"
    Amber = "#FF6F00"
    Orange = "#E65100"
    DeepOrange = "#BF360C"
    Brown = "#3E2723"
    Grey = "#212121"
    BlueGrey = "#263238"


class DarkThemeColors:

    Red = "#F44336"
    Pink = "#F48FB1"
    Purple = "#CE93D8"
    DeepPurple = "#B39DDB"
    Indigo = "#9FA8DA"
    Blue = "#90CAF9"
    LightBlue = "#81D4FA"
    Cyan = "#80DEEA"
    Teal = "#80CBC4"
    Green = "#A5D6A7"
    LightGreen = "#C5E1A5"
    Lime = "#E6EE9C"
    Yellow = "#FFF59D"
    Amber = "#FFE082"
    Orange = "#FFCC80"
    DeepOrange = "#FFAB91"
    Brown = "#BCAAA4"
    Grey = "#EEEEEE"
    BlueGrey = "#B0BEC5"


LIGHT_STYLES = {
    'keyword': format(LightThemeColors.Blue, 'bold'),
    'operator': format(LightThemeColors.Red, 'bold'),
    'brace': format(LightThemeColors.Purple),
    'defclass': format(LightThemeColors.Indigo, 'bold'),
    'string': format(LightThemeColors.Amber),
    'string2': format(LightThemeColors.DeepPurple),
    'comment': format(LightThemeColors.Green, 'italic'),
    'self': format(LightThemeColors.Blue, 'bold'),
    'numbers': format(LightThemeColors.Teal),
}


DARK_STYLES = {
    'keyword': format(DarkThemeColors.Blue, 'bold'),
    'operator': format(DarkThemeColors.Red, 'bold'),
    'brace': format(DarkThemeColors.Purple),
    'defclass': format(DarkThemeColors.Indigo, 'bold'),
    'string': format(DarkThemeColors.Amber),
    'string2': format(DarkThemeColors.DeepPurple),
    'comment': format(DarkThemeColors.Green, 'italic'),
    'self': format(DarkThemeColors.Blue, 'bold'),
    'numbers': format(DarkThemeColors.Teal),
}


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language.
    """
    # Python keywords
    keywords = [
        'and', 'assert', 'break', 'class', 'continue', 'def',
        'del', 'elif', 'else', 'except', 'exec', 'finally',
        'for', 'from', 'global', 'if', 'import', 'in',
        'is', 'lambda', 'not', 'or', 'pass', 'print',
        'raise', 'return', 'try', 'while', 'yield',
        'None', 'True', 'False', 'async', 'await',
    ]

    # Python operators
    operators = [
        '=',
        # Comparison
        '==', '!=', '<', '<=', '>', '>=',
        # Arithmetic
        '\+', '-', '\*', '/', '//', '\%', '\*\*',
        # In-place
        '\+=', '-=', '\*=', '/=', '\%=',
        # Bitwise
        '\^', '\|', '\&', '\~', '>>', '<<',
    ]

    # Python braces
    braces = [
        '\{', '\}', '\(', '\)', '\[', '\]',
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        # Multi-line strings (expression, flag, style)
        # FIXME: The triple-quotes in these two lines will mess up the
        # syntax highlighting from this point onward
        self.tri_single = (QRegExp("'''"), 1, 'string2')
        self.tri_double = (QRegExp('"""'), 2, 'string2')

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r'\b%s\b' % w, 0, 'keyword')
                  for w in PythonHighlighter.keywords]
        rules += [(r'%s' % o, 0, 'operator')
                  for o in PythonHighlighter.operators]
        rules += [(r'%s' % b, 0, 'brace')
                  for b in PythonHighlighter.braces]

        # All other rules
        rules += [

            # 'self'
            (r'\bself\b', 0, 'self'),

            # 'def' followed by an identifier
            (r'\bdef\b\s*(\w+)', 1, 'defclass'),
            # 'class' followed by an identifier
            (r'\bclass\b\s*(\w+)', 1, 'defclass'),

            # Numeric literals
            (r'\b[+-]?[0-9]+[lL]?\b', 0, 'numbers'),
            (r'\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b', 0, 'numbers'),
            (r'\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b', 0, 'numbers'),

            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, 'string'),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, 'string'),

            # From '#' until a newline
            (r'#[^\n]*', 0, 'comment'),

        ]

        # Build a QRegExp for each pattern
        self.rules = [(QRegExp(pat), index, fmt)
                      for (pat, index, fmt) in rules]

    @property
    def styles(self):
        app = QtGui.QApplication.instance()
        return DARK_STYLES if app.dark_mode else LIGHT_STYLES

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)
            format = self.styles[format]

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, self.styles[style])
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False



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
        self.hl = PythonHighlighter(self.ui.codeView.document())
        app = QtGui.QApplication.instance()
        if QT_LIB in ['PyQt5', 'PySide2']:
            # Qt4 does not have a paletteChanged signal
            app.paletteChanged.connect(self.updateTheme)
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

    def updateTheme(self):
        self.hl = PythonHighlighter(self.ui.codeView.document())

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


def main():
    app = pg.mkQApp()
    loader = ExampleLoader()
    app.exec_()

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()