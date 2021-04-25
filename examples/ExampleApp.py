import keyword
import os
import re
import sys
import subprocess
from argparse import Namespace
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
from collections import OrderedDict
from .utils import examples

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)
app = pg.mkQApp()

import importlib
ui_template = importlib.import_module(
    f'exampleLoaderTemplate_{QT_LIB.lower()}')


# based on https://github.com/art1415926535/PyQt5-syntax-highlighting

QRegularExpression = QtCore.QRegularExpression

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
    keywords = keyword.kwlist

    # Python operators
    operators = [
        r'=',
        # Comparison
        r'==', r'!=', r'<', r'<=', r'>', r'>=',
        # Arithmetic
        r'\+', r"-", r'\*', r'/', r'//', r'%', r'\*\*',
        # In-place
        r'\+=', r'-=', r'\*=', r'/=', r'\%=',
        # Bitwise
        r'\^', r'\|', r'&', r'~', r'>>', r'<<',
    ]

    # Python braces
    braces = [
        r'\{', r'\}', r'\(', r'\)', r'\[', r'\]',
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        # Multi-line strings (expression, flag, style)
        self.tri_single = (QRegularExpression("'''"), 1, 'string2')
        self.tri_double = (QRegularExpression('"""'), 2, 'string2')

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r'\b%s\b' % w, 0, 'keyword')
                  for w in PythonHighlighter.keywords]
        rules += [(o, 0, 'operator')
                  for o in PythonHighlighter.operators]
        rules += [(b, 0, 'brace')
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
        self.rules = rules

    @property
    def styles(self):
        app = QtGui.QApplication.instance()
        return DARK_STYLES if app.property('darkMode') else LIGHT_STYLES

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            format = self.styles[format]

            for n, match in enumerate(re.finditer(expression, text)):
                if n < nth:
                    continue
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)
        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. 
        
        =========== ==========================================================
        delimiter   (QRegularExpression) for triple-single-quotes or 
                    triple-double-quotes
        in_state    (int) to represent the corresponding state changes when 
                    inside those strings. Returns True if we're still inside a
                    multi-line string when this function is finished.
        style       (str) representation of the kind of style to use
        =========== ==========================================================
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            match = delimiter.match(text)
            start = match.capturedStart()
            # Move past this match
            add = match.capturedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            match = delimiter.match(text, start + add)
            end = match.capturedEnd()
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + match.capturedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, self.styles[style])
            # Look for the next match
            match = delimiter.match(text, start + length)
            start = match.capturedStart()

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False



class ExampleLoader(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = ui_template.Ui_Form()
        self.cw = QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.ui.setupUi(self.cw)
        self.setWindowTitle("PyQtGraph Examples")
        self.codeBtn = QtGui.QPushButton('Run Edited Code')
        self.codeLayout = QtGui.QGridLayout()
        self.ui.codeView.setLayout(self.codeLayout)
        self.hl = PythonHighlighter(self.ui.codeView.document())
        app = QtGui.QApplication.instance()
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
        app.setProperty('darkMode', True)

    def updateTheme(self):
        self.hl = PythonHighlighter(self.ui.codeView.document())

    def populateTree(self, root, examples):
        bold_font = None
        for key, val in examples.items():
            item = QtGui.QTreeWidgetItem([key])
            self.itemCache.append(item) # PyQt 4.9.6 no longer keeps references to these wrappers,
                                        # so we need to make an explicit reference or else the .file
                                        # attribute will disappear.
            if isinstance(val, OrderedDict):
                self.populateTree(item, val)
            elif isinstance(val, Namespace):
                item.file = val.filename
                if 'recommended' in val:
                    if bold_font is None:
                        bold_font = item.font(0)
                        bold_font.setBold(True)
                    item.setFont(0, bold_font)
            else:
                item.file = val
            root.addChild(item)

    def currentFile(self):
        item = self.ui.exampleTree.currentItem()
        if hasattr(item, 'file'):
            global path
            return os.path.join(path, item.file)
        return None

    def loadFile(self, edited=False):

        qtLib = str(self.ui.qtLibCombo.currentText())

        env = None
        if qtLib != 'default':
            env = dict(os.environ, PYQTGRAPH_QT_LIB=qtLib)

        if edited:
            path = os.path.abspath(os.path.dirname(__file__))
            proc = subprocess.Popen([sys.executable, '-'], stdin=subprocess.PIPE, cwd=path, env=env)
            code = str(self.ui.codeView.toPlainText()).encode('UTF-8')
            proc.stdin.write(code)
            proc.stdin.close()
        else:
            fn = self.currentFile()
            if fn is None:
                return
            subprocess.Popen([sys.executable, fn], env=env)

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
    main()
