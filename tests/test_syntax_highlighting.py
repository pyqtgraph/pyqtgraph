import glob
import os

import pyqtgraph as pg
from pyqtgraph.examples.syntax import PythonHighlighter
from pyqtgraph.Qt import QtGui

pg.mkQApp()

# `styles` is a property that picks the light or dark palette from the
# application's `darkMode` at each access, so expected colors must come from it
# rather than from LIGHT_STYLES directly -- otherwise every test in this module
# fails when run on a dark desktop.
_styleProbeDocument = QtGui.QTextDocument()
_styleProbe = PythonHighlighter(_styleProbeDocument)


def highlight(text):
    """Highlight `text` and return the QTextDocument it was applied to.

    The highlighter is parented to the document and `rehighlight` writes the
    formats into the block layouts synchronously, so the document alone is
    enough to inspect the result. The parenting matters: `~QSyntaxHighlighter`
    clears every block's formats, so an unparented highlighter would take the
    results with it when collected.
    """
    document = QtGui.QTextDocument()
    document.setPlainText(text)
    PythonHighlighter(document).rehighlight()
    return document


def formatAt(document, position):
    """Return the QTextCharFormat applied at a character position, or None."""
    block = document.findBlock(position)
    offset = position - block.position()
    for fmtRange in block.layout().formats():
        if fmtRange.start <= offset < fmtRange.start + fmtRange.length:
            return fmtRange.format
    return None


def colorAt(document, position):
    """Return the foreground color name applied at a position, or None."""
    fmt = formatAt(document, position)
    return None if fmt is None else fmt.foreground().color().name()


def styleColor(name):
    """Return the foreground color name of the named style.

    Note that 'keyword' and 'self' share a color, so they cannot be told apart
    by color alone; every other style used here is unique.
    """
    return _styleProbe.styles[name].foreground().color().name()


def assertSpanColor(document, start, end, styleName):
    """Assert every position in [start, end) has `styleName`'s color."""
    expected = styleColor(styleName)
    actual = {pos: colorAt(document, pos) for pos in range(start, end)}
    assert set(actual.values()) == {expected}, actual


# -- the bug: a '#' inside a string swallowed the real trailing comment -------

def test_trailing_comment_after_string_containing_hash():
    text = 'x = "value # not a comment" # real comment'
    document = highlight(text)

    assertSpanColor(document, text.index('"'), text.index('"') + 23, 'string')
    assertSpanColor(document, text.rindex('#'), len(text), 'comment')


def test_keyword_inside_trailing_comment_is_not_highlighted():
    text = 'x = "value # not a comment" # this is a real comment'
    document = highlight(text)

    # ' is ' is a keyword everywhere except inside a comment
    assert colorAt(document, text.index(' is ') + 1) == styleColor('comment')


def test_apostrophes_in_comment_do_not_break_it():
    """The form this bug actually takes in the example corpus.

    The apostrophes in "old's ... new's" are read as a single-quoted string,
    which used to suppress the comment they sit inside.
    """
    text = "old.resetBrush()  # reset old's brush before setting new's"
    document = highlight(text)

    assertSpanColor(document, text.index('#'), len(text), 'comment')


def test_quotes_inside_comment_do_not_break_it():
    text = 'x = 5  # comment with "quotes # inside"'
    document = highlight(text)

    assertSpanColor(document, text.index('#'), len(text), 'comment')


def test_comment_after_two_strings():
    text = 'x = "a # b" + "c # d" # real'
    document = highlight(text)

    # the comment must start at the last '#', not at either string's '#'
    assert colorAt(document, text.index('# b')) == styleColor('string')
    assert colorAt(document, text.index('# d')) == styleColor('string')
    assertSpanColor(document, text.rindex('#'), len(text), 'comment')


def test_comment_after_adjacent_strings():
    text = 'x = "a" "b # c" # real'
    document = highlight(text)

    assert colorAt(document, text.index('# c')) == styleColor('string')
    assertSpanColor(document, text.rindex('#'), len(text), 'comment')


def test_hash_only_inside_string_starts_no_comment():
    text = 'x = "a # b"'
    document = highlight(text)

    assertSpanColor(document, text.index('"'), len(text), 'string')


def test_comment_immediately_after_closing_quote():
    text = 'x = "a"# tight'
    document = highlight(text)

    assertSpanColor(document, text.index('"'), text.index('#'), 'string')
    assertSpanColor(document, text.index('#'), len(text), 'comment')


# -- comments and strings in isolation ---------------------------------------

def test_plain_trailing_comment():
    text = 'x = 5  # a comment'
    document = highlight(text)

    assertSpanColor(document, text.index('#'), len(text), 'comment')


def test_full_line_comment():
    text = '    # a full line comment'
    document = highlight(text)

    assertSpanColor(document, text.index('#'), len(text), 'comment')


def test_keyword_inside_full_line_comment_is_not_highlighted():
    text = '# if this were code'
    document = highlight(text)

    assertSpanColor(document, 0, len(text), 'comment')


def test_triple_quote_in_full_line_comment_does_not_open_a_string():
    """A full-line comment is exempt from multi-line string detection.

    Without that exemption the ''' below would open a string and every
    following line would render as string content.
    """
    text = '\n'.join(["# use ''' for docstrings", 'x = 1'])
    document = highlight(text)

    assert colorAt(document, text.rindex('1')) == styleColor('numbers')


def test_full_line_comment_clears_a_stray_multi_line_string_state():
    """A full-line comment resets the block state, healing a false string.

    The single-line string rule cannot see that the ''' below is string
    content, so `match_multiline` opens a multi-line string that never closes.
    The fast path's unconditional `setCurrentBlockState(0)` is what stops that
    from running to end of file, which is why guarding the fast path on
    `previousBlockState()` is a regression rather than a cleanup.
    """
    text = '\n'.join(['x = "\'\'\'"', '# a full line comment', 'y = 1'])
    document = highlight(text)

    start = text.index('#')
    assertSpanColor(document, start, start + len('# a full line comment'),
                    'comment')
    assert colorAt(document, text.rindex('1')) == styleColor('numbers')


def test_double_quoted_string():
    text = 'x = "plain string"'
    document = highlight(text)

    assertSpanColor(document, text.index('"'), len(text), 'string')


def test_single_quoted_string():
    text = "x = 'plain string'"
    document = highlight(text)

    assertSpanColor(document, text.index("'"), len(text), 'string')


# -- other rules still apply -------------------------------------------------

def test_keyword_outside_string_is_highlighted():
    text = 'if x:'
    document = highlight(text)

    assertSpanColor(document, 0, 2, 'keyword')


def test_keyword_inside_string_is_not_highlighted():
    text = 'x = "if statement"'
    document = highlight(text)

    assert colorAt(document, text.index('if')) == styleColor('string')


def test_operator_is_highlighted():
    text = 'a + b'
    document = highlight(text)

    assert colorAt(document, text.index('+')) == styleColor('operator')


def test_number_is_highlighted():
    text = 'x = 42'
    document = highlight(text)

    assertSpanColor(document, text.index('42'), len(text), 'numbers')


def test_braces_are_highlighted():
    text = 'func()'
    document = highlight(text)

    assertSpanColor(document, text.index('('), len(text), 'brace')


# -- residual known limitations, guarded against the shipped corpus ----------

def test_no_shipped_example_has_a_trailing_comment_that_is_cut_short():
    """Every trailing comment must render as comment color through to end
    of line (the spec's Trailing comment scenario). This is also the
    observable shape of a documented residual limitation: a `'''` or
    `\"\"\"` inside a *trailing* (non-full-line) comment opens a spurious
    multi-line string via `match_multiline`, which overwrites everything
    from the delimiter onward with string color instead of comment color --
    so the comment run ends before end of line rather than containing the
    delimiter literally. No shipped example hits this today, and the
    design's regression claim (rendering is byte-identical to master apart
    from this one exempted case) depends on that staying true. This scans
    the corpus so a future example that introduces the pattern fails here
    instead of silently mis-rendering in the example browser.
    """
    examplesDir = os.path.join(os.path.dirname(pg.__file__), 'examples')
    paths = sorted(glob.glob(os.path.join(examplesDir, '**', '*.py'), recursive=True))
    assert paths, 'expected to find shipped example scripts'

    commentColor = styleColor('comment')
    offenders = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            text = f.read()
        document = highlight(text)
        block = document.begin()
        while block.isValid():
            blockText = block.text()
            if blockText and not blockText.lstrip().startswith('#'):
                commentPositions = [
                    i for i in range(len(blockText))
                    if colorAt(document, block.position() + i) == commentColor
                ]
                if commentPositions and commentPositions[-1] != len(blockText) - 1:
                    offenders.append(f'{path}: {blockText!r}')
            block = block.next()

    assert offenders == []
