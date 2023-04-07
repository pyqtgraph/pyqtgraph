import keyword
import os
import pkgutil
import re
import subprocess
import sys
from argparse import Namespace
from collections import OrderedDict
from functools import lru_cache

import pyqtgraph as pg
from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets

app = pg.mkQApp()


path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

import exampleLoaderTemplate_generic as ui_template
import utils

# based on https://github.com/art1415926535/PyQt5-syntax-highlighting

QRegularExpression = QtCore.QRegularExpression

QFont = QtGui.QFont
QColor = QtGui.QColor
QTextCharFormat = QtGui.QTextCharFormat
QSyntaxHighlighter = QtGui.QSyntaxHighlighter


def charFormat(color, style='', background=None):
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
        _format.setFontWeight(QFont.Weight.Bold)
    if 'italic' in style:
        _format.setFontItalic(True)
    if background is not None:
        _format.setBackground(pg.mkColor(background))

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
    'keyword': charFormat(LightThemeColors.Blue, 'bold'),
    'operator': charFormat(LightThemeColors.Red, 'bold'),
    'brace': charFormat(LightThemeColors.Purple),
    'defclass': charFormat(LightThemeColors.Indigo, 'bold'),
    'string': charFormat(LightThemeColors.Amber),
    'string2': charFormat(LightThemeColors.DeepPurple),
    'comment': charFormat(LightThemeColors.Green, 'italic'),
    'self': charFormat(LightThemeColors.Blue, 'bold'),
    'numbers': charFormat(LightThemeColors.Teal),
}


DARK_STYLES = {
    'keyword': charFormat(DarkThemeColors.Blue, 'bold'),
    'operator': charFormat(DarkThemeColors.Red, 'bold'),
    'brace': charFormat(DarkThemeColors.Purple),
    'defclass': charFormat(DarkThemeColors.Indigo, 'bold'),
    'string': charFormat(DarkThemeColors.Amber),
    'string2': charFormat(DarkThemeColors.DeepPurple),
    'comment': charFormat(DarkThemeColors.Green, 'italic'),
    'self': charFormat(DarkThemeColors.Blue, 'bold'),
    'numbers': charFormat(DarkThemeColors.Teal),
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
        super().__init__(document)

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
        self.searchText = None

    @property
    def styles(self):
        app = QtWidgets.QApplication.instance()
        return DARK_STYLES if app.property('darkMode') else LIGHT_STYLES

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        # Do other syntax formatting
        rules = self.rules.copy()
        for expression, nth, format in rules:
            format = self.styles[format]

            for n, match in enumerate(re.finditer(expression, text)):
                if n < nth:
                    continue
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)

        self.applySearchHighlight(text)
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
            # Highlighting sits on top of this formatting
            # Look for the next match
            match = delimiter.match(text, start + length)
            start = match.capturedStart()

        self.applySearchHighlight(text)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False

    def applySearchHighlight(self, text):
        if not self.searchText:
            return
        expr = f'(?i){self.searchText}'
        palette: QtGui.QPalette = app.palette()
        color = palette.highlight().color()
        fgndColor = palette.color(palette.ColorGroup.Current,
                                  palette.ColorRole.Text).name()
        style = charFormat(fgndColor, background=color.name())
        for match in re.finditer(expr, text):
            start = match.start()
            length = match.end() - start
            self.setFormat(start, length, style)


def unnestedDict(exDict):
    """Converts a dict-of-dicts to a singly nested dict for non-recursive parsing"""
    out = {}
    for kk, vv in exDict.items():
        if isinstance(vv, dict):
            out.update(unnestedDict(vv))
        else:
            out[kk] = vv
    return out



class ExampleLoader(QtWidgets.QMainWindow):
    # update qtLibCombo item order to match bindings in the UI file and recreate
    # the templates files if you change bindings.
    bindings = {'PyQt6': 0, 'PySide6': 1, 'PyQt5': 2, 'PySide2': 3}
    modules = tuple(m.name for m in pkgutil.iter_modules())
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = ui_template.Ui_Form()
        self.cw = QtWidgets.QWidget()
        self.setCentralWidget(self.cw)
        self.ui.setupUi(self.cw)
        self.setWindowTitle("PyQtGraph Examples")
        self.codeBtn = QtWidgets.QPushButton('Run Edited Code')
        self.codeLayout = QtWidgets.QGridLayout()
        self.ui.codeView.setLayout(self.codeLayout)
        self.hl = PythonHighlighter(self.ui.codeView.document())
        app = QtWidgets.QApplication.instance()
        app.paletteChanged.connect(self.updateTheme)
        policy = QtWidgets.QSizePolicy.Policy.Expanding
        self.codeLayout.addItem(QtWidgets.QSpacerItem(100,100, policy, policy), 0, 0)
        self.codeLayout.addWidget(self.codeBtn, 1, 1)
        self.codeBtn.hide()

        textFil = self.ui.exampleFilter
        self.curListener = None
        self.ui.exampleFilter.setFocus()
        self.ui.qtLibCombo.addItems(self.bindings.keys())
        self.ui.qtLibCombo.setCurrentIndex(self.bindings[QT_LIB])


        def onComboChanged(searchType):
            if self.curListener is not None:
                self.curListener.disconnect()
            self.curListener = textFil.textChanged
            # In case the regex was invalid before switching to title search,
            # ensure the "invalid" color is reset
            self.ui.exampleFilter.setStyleSheet('')
            if searchType == 'Content Search':
                self.curListener.connect(self.filterByContent)
            else:
                self.hl.searchText = None
                self.curListener.connect(self.filterByTitle)
            # Fire on current text, too
            self.curListener.emit(textFil.text())

        self.ui.searchFiles.currentTextChanged.connect(onComboChanged)
        onComboChanged(self.ui.searchFiles.currentText())

        self.itemCache = []
        self.populateTree(self.ui.exampleTree.invisibleRootItem(), utils.examples_)
        self.ui.exampleTree.expandAll()

        self.resize(1000,500)
        self.show()
        self.ui.splitter.setSizes([250,750])

        self.oldText = self.ui.codeView.toPlainText()
        self.ui.loadBtn.clicked.connect(self.loadFile)
        self.ui.exampleTree.currentItemChanged.connect(self.showFile)
        self.ui.exampleTree.itemDoubleClicked.connect(self.loadFile)
        self.ui.codeView.textChanged.connect(self.onTextChange)
        self.codeBtn.clicked.connect(self.runEditedCode)
        self.updateCodeViewTabWidth(self.ui.codeView.font())

    def updateCodeViewTabWidth(self,font):
        """
        Change the codeView tabStopDistance to 4 spaces based on the size of the current font
        """
        fm = QtGui.QFontMetrics(font)
        tabWidth = fm.horizontalAdvance(' ' * 4)
        # the default value is 80 pixels! that's more than 2x what we want.
        self.ui.codeView.setTabStopDistance(tabWidth)

    def showEvent(self, event) -> None:
        super(ExampleLoader, self).showEvent(event)
        disabledColor = QColor(QtCore.Qt.GlobalColor.red)
        for name, idx in self.bindings.items():
            disableBinding = name not in self.modules
            if disableBinding:
                item = self.ui.qtLibCombo.model().item(idx)
                item.setData(disabledColor, QtCore.Qt.ItemDataRole.ForegroundRole)
                item.setEnabled(False)
                item.setToolTip(f'{item.text()} is not installed')

    def onTextChange(self):
        """
        textChanged fires when the highlighter is reassigned the same document.
        Prevent this from showing "run edited code" by checking for actual
        content change
        """
        newText = self.ui.codeView.toPlainText()
        if newText != self.oldText:
            self.oldText = newText
            self.codeEdited() 

    def filterByTitle(self, text):
        self.showExamplesByTitle(self.getMatchingTitles(text))
        self.hl.setDocument(self.ui.codeView.document())

    def filterByContent(self, text=None):
        # If the new text isn't valid regex, fail early and highlight the search filter red to indicate a problem
        # to the user
        validRegex = True
        try:
            re.compile(text)
            self.ui.exampleFilter.setStyleSheet('')
        except re.error:
            colors = DarkThemeColors if app.property('darkMode') else LightThemeColors
            errorColor = pg.mkColor(colors.Red)
            validRegex = False
            errorColor.setAlpha(100)
            # Tuple prints nicely :)
            self.ui.exampleFilter.setStyleSheet(f'background: rgba{errorColor.getRgb()}')
        if not validRegex:
            return
        checkDict = unnestedDict(utils.examples_)
        self.hl.searchText = text
        # Need to reapply to current document
        self.hl.setDocument(self.ui.codeView.document())
        titles = []
        text = text.lower()
        for kk, vv in checkDict.items():
            if isinstance(vv, Namespace):
                vv = vv.filename
            filename = os.path.join(path, vv)
            contents = self.getExampleContent(filename).lower()
            if text in contents:
                titles.append(kk)
        self.showExamplesByTitle(titles)

    def getMatchingTitles(self, text, exDict=None, acceptAll=False):
        if exDict is None:
            exDict = utils.examples_
        text = text.lower()
        titles = []
        for kk, vv in exDict.items():
            matched = acceptAll or text in kk.lower()
            if isinstance(vv, dict):
                titles.extend(self.getMatchingTitles(text, vv, acceptAll=matched))
            elif matched:
                titles.append(kk)
        return titles

    def showExamplesByTitle(self, titles):
        QTWI = QtWidgets.QTreeWidgetItemIterator
        flag = QTWI.IteratorFlag.NoChildren
        treeIter = QTWI(self.ui.exampleTree, flag)
        item = treeIter.value()
        while item is not None:
            parent = item.parent()
            show = (item.childCount() or item.text(0) in titles)
            item.setHidden(not show)

            # If all children of a parent are gone, hide it
            if parent:
                hideParent = True
                for ii in range(parent.childCount()):
                    if not parent.child(ii).isHidden():
                        hideParent = False
                        break
                parent.setHidden(hideParent)

            treeIter += 1
            item = treeIter.value()

    def simulate_black_mode(self):
        """
        used to simulate MacOS "black mode" on other platforms
        intended for debug only, as it manage only the QPlainTextEdit
        """
        # first, a dark background
        c = QtGui.QColor('#171717')
        p = self.ui.codeView.palette()
        p.setColor(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Base, c)
        p.setColor(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Base, c)
        self.ui.codeView.setPalette(p)
        # then, a light font
        f = QtGui.QTextCharFormat()
        f.setForeground(QtGui.QColor('white'))
        self.ui.codeView.setCurrentCharFormat(f)
        # finally, override application automatic detection
        app = QtWidgets.QApplication.instance()
        app.setProperty('darkMode', True)

    def updateTheme(self):
        self.hl = PythonHighlighter(self.ui.codeView.document())

    def populateTree(self, root, examples):
        bold_font = None
        for key, val in examples.items():
            item = QtWidgets.QTreeWidgetItem([key])
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
            return os.path.join(path, item.file)
        return None

    def loadFile(self, *, edited=False):
        # make *edited* keyword-only so it is not confused for extra arguments
        # sent by ui signals
        qtLib = self.ui.qtLibCombo.currentText()
        env = dict(os.environ, PYQTGRAPH_QT_LIB=qtLib)
        example_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.dirname(os.path.dirname(example_path))
        env['PYTHONPATH'] = f'{path}'
        if edited:
            proc = subprocess.Popen([sys.executable, '-'], stdin=subprocess.PIPE, cwd=example_path, env=env)
            code = self.ui.codeView.toPlainText().encode('UTF-8')
            proc.stdin.write(code)
            proc.stdin.close()
        else:
            fn = self.currentFile()
            if fn is None:
                return
            subprocess.Popen([sys.executable, fn], cwd=path, env=env)

    def showFile(self):
        fn = self.currentFile()
        text = self.getExampleContent(fn)
        self.ui.codeView.setPlainText(text)
        self.ui.loadedFileLabel.setText(fn)
        self.codeBtn.hide()

    @lru_cache(100)
    def getExampleContent(self, filename):
        if filename is None:
            self.ui.codeView.clear()
            return
        if os.path.isdir(filename):
            filename = os.path.join(filename, '__main__.py')
        with open(filename, "r") as currentFile:
            text = currentFile.read()
        return text

    def codeEdited(self):
        self.codeBtn.show()

    def runEditedCode(self):
        self.loadFile(edited=True)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if not (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            return
        key = event.key()
        Key = QtCore.Qt.Key

        # Allow quick navigate to search
        if key == Key.Key_F:
            self.ui.exampleFilter.setFocus()
            event.accept()
            return

        if key not in [Key.Key_Plus, Key.Key_Minus, Key.Key_Underscore, Key.Key_Equal, Key.Key_0]:
            return
        font = self.ui.codeView.font()
        oldSize = font.pointSize()
        if key == Key.Key_Plus or key == Key.Key_Equal:
            font.setPointSize(oldSize + max(oldSize*.15, 1))
        elif key == Key.Key_Minus or key == Key.Key_Underscore:
            newSize = oldSize - max(oldSize*.15, 1)
            font.setPointSize(max(newSize, 1))
        elif key == Key.Key_0:
            # Reset to original size
            font.setPointSize(10)
        self.ui.codeView.setFont(font)
        self.updateCodeViewTabWidth(font)
        event.accept()

def main():
    app = pg.mkQApp()
    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(
        loader.ui.exampleTree.model().index(0,0)
    )
    pg.exec()

if __name__ == '__main__':
    main()
