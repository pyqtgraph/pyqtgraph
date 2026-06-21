# based on https://github.com/art1415926535/PyQt5-syntax-highlighting

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import keyword
import re

QRegularExpression = QtCore.QRegularExpression

QFont = QtGui.QFont
QColor = QtGui.QColor
QTextCharFormat = QtGui.QTextCharFormat
QSyntaxHighlighter = QtGui.QSyntaxHighlighter


def charFormat(color, style='', background=None):
    """
    Return a QTextCharFormat with the given attributes.
    """
    _color = pg.functions.mkColor(color)

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
        """Apply syntax highlighting to the given block of text."""

        rules = self.rules.copy()
        string_spans = []

        # Special case: full-line comment
        stripped = text.lstrip()
        if stripped.startswith('#'):
            start = len(text) - len(stripped)
            self.setFormat(start, len(text) - start, self.styles['comment'])
            self.setCurrentBlockState(0)
            self.applySearchHighlight(text)
            return

        # Next: apply string rules and record spans
        for expression, nth, format in rules:
            if format not in ('string', 'string2'):
                continue
            format = self.styles[format]
            for n, match in enumerate(re.finditer(expression, text)):
                if n < nth:
                    continue
                start, end = match.span()
                self.setFormat(start, end - start, format)
                string_spans.append((start, end))

        # Then: apply other rules only if not in a string
        for expression, nth, format in rules:
            if format in ('string', 'string2'):
                continue
            format = self.styles[format]
            for n, match in enumerate(re.finditer(expression, text)):
                if n < nth:
                    continue
                start, end = match.span()
                if any(start < e and end > s for s, e in string_spans):
                    continue  # Skip overlapping with string
                self.setFormat(start, end - start, format)

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
        app = QtWidgets.QApplication.instance()
        palette: QtGui.QPalette = app.palette()
        color = palette.highlight().color()
        fgndColor = palette.color(palette.ColorGroup.Current,
                                  palette.ColorRole.Text).name()
        style = charFormat(fgndColor, background=color.name())
        for match in re.finditer(expr, text):
            start = match.start()
            length = match.end() - start
            self.setFormat(start, length, style)
