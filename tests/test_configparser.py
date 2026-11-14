import datetime
import os
from collections import OrderedDict

import numpy as np
import pytest

from pyqtgraph import configfile
from pyqtgraph.colormap import ColorMap
from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtCore


def test_longArrays(tmpdir):
    """
    Test config saving and loading of long arrays.
    """
    arr = np.arange(20)

    tf = tmpdir.join("config.cfg")
    configfile.writeConfigFile({'arr': arr}, tf)
    config = configfile.readConfigFile(tf)
    assert all(config['arr'] == arr)


def test_multipleParameters(tmpdir):
    """
    Test config saving and loading of multiple parameters.
    """

    par1 = [1,2,3]
    par2 = "Test"
    par3 = {'a':3,'b':'c'}

    tf = tmpdir.join("config.cfg")
    configfile.writeConfigFile({'par1':par1, 'par2':par2, 'par3':par3}, tf)
    config = configfile.readConfigFile(tf)

    assert config['par1'] == par1
    assert config['par2'] == par2
    assert config['par3'] == par3


def test_duplicate_keys_error(tmpdir):
    """
    Test that an error is raised when duplicate keys are present in the config file.
    """

    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write('a: 1\n')
        f.write('a: 2\n')

    try:
        configfile.readConfigFile(tf)
    except configfile.ParseError as e:
        assert 'Duplicate key' in str(e)
    else:
        assert False, "Expected ParseError"


def test_line_numbers_account_for_comments_and_blanks(tmpdir):
    """
    Test that line numbers in ParseError account for comments and blank lines.
    """

    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write('a: 1\n')
        f.write('\n')
        f.write('# comment\n')
        f.write('a: 2\n')

    try:
        configfile.readConfigFile(tf)
    except configfile.ParseError as e:
        assert 'at line 4' in str(e)
    else:
        assert False, "Expected ParseError"


def test_comment_indentation_is_ignored(tmpdir):
    """
    Test that comment indentation is ignored.
    """

    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write('a:\n')
        f.write('        # comment\n')
        f.write('    b:\n')
        f.write('# more comments\n')
        f.write('        c: 2\n')

    retval = configfile.readConfigFile(tf)
    assert retval['a']['b']['c'] == 2


def test_malicious_value_is_not_executed(tmpdir):
    """
    Test that a crafted config value cannot execute code (issue #3702).

    The proof-of-concept payload from the issue must raise ParseError and
    must not create the marker file it tries to write.
    """
    marker = tmpdir.join("pwned.txt")
    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write(f"note: __import__('os').system('touch {marker}')\n")

    with pytest.raises(configfile.ParseError):
        configfile.readConfigFile(tf)

    assert not os.path.exists(marker)


@pytest.mark.parametrize("payload", [
    "__import__('os').system('id')",
    "open('/etc/passwd').read()",
    "().__class__.__base__.__subclasses__()",
    "eval('1+1')",
    "getattr(array, '__class__')",
    "array.__class__",
    "[x for x in range(3)]",
    "(lambda: 1)()",
    "1 if True else 2",
    "{'a': 1}['a']",
    "Point(1, 2).x()",
    "datetime.datetime.now()",
    "QtCore.QProcess.execute('true')",
])
def test_disallowed_expressions_raise_parse_error(tmpdir, payload):
    """
    Test that expressions outside the restricted config grammar are rejected.
    """
    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write(f"x: {payload}\n")

    with pytest.raises(configfile.ParseError):
        configfile.readConfigFile(tf)


def test_malicious_tuple_key_is_not_executed(tmpdir):
    """
    Test that a crafted tuple-like key cannot execute code (issue #3702).

    Keys that look like tuples are evaluated; a payload key must fall back
    to the plain string key without executing anything.
    """
    marker = tmpdir.join("pwned.txt")
    key = f"(__import__('os').system('touch {marker}'),)"
    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write(f"{key}: 1\n")

    config = configfile.readConfigFile(tf)

    assert not os.path.exists(marker)
    assert key in config
    assert config[key] == 1


def test_legitimate_expressions_still_parse(tmpdir):
    """
    Test that the restricted grammar still accepts the value forms that
    config files legitimately contain: literals, unit arithmetic, and
    calls to the allow-listed names and classes.
    """
    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write(
            "a: 1\n"
            "b: -2.5\n"
            "c: 1e-3\n"
            "d: 'hello'\n"
            "e: True\n"
            "f: None\n"
            "g: [1, 2, 3]\n"
            "h: (1, 2)\n"
            "i: {'x': 1}\n"
            "j: {1, 2}\n"
            "k: 1+2j\n"
            "l: 20*pA\n"
            "m: datetime.datetime(2026, 1, 2, 3, 4, 5)\n"
            "n: array([1, 2, 3], dtype=float32)\n"
            "o: Point(1.5, 2.5)\n"
            "p: OrderedDict([('a', 1), ('b', 2)])\n"
            "q: QtCore.Qt.AlignmentFlag.AlignLeft\n"
            "r: QtCore.QPointF(1.5, 2.5)\n"
            "s: QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop\n"
            "t: float64(3)\n"
            "u: 2**10\n"
            "v: b'bytes'\n"
            "w: 0x10\n"
            "(3, 4): 5\n"
        )

    config = configfile.readConfigFile(tf)

    assert config['a'] == 1
    assert config['b'] == -2.5
    assert config['c'] == 1e-3
    assert config['d'] == 'hello'
    assert config['e'] is True
    assert config['f'] is None
    assert config['g'] == [1, 2, 3]
    assert config['h'] == (1, 2)
    assert config['i'] == {'x': 1}
    assert config['j'] == {1, 2}
    assert config['k'] == 1 + 2j
    assert config['l'] == 20e-12
    assert config['m'] == datetime.datetime(2026, 1, 2, 3, 4, 5)
    assert config['n'].dtype == np.float32
    assert list(config['n']) == [1, 2, 3]
    assert config['o'] == Point(1.5, 2.5)
    assert config['p'] == OrderedDict([('a', 1), ('b', 2)])
    assert config['q'] == QtCore.Qt.AlignmentFlag.AlignLeft
    assert config['r'] == QtCore.QPointF(1.5, 2.5)
    assert config['s'] == (
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
    )
    assert config['t'] == np.float64(3)
    assert config['u'] == 1024
    assert config['v'] == b'bytes'
    assert config['w'] == 16
    assert config[(3, 4)] == 5


def test_roundtrip_of_allow_listed_types(tmpdir):
    """
    Test that writeConfigFile/readConfigFile round-trips the types whose
    repr forms rely on the allow-listed names and classes.
    """
    cmap = ColorMap([0.0, 1.0], [(0, 0, 0), (255, 0, 0)])
    data = {
        'arr1d': np.arange(5),
        'arr2d': np.arange(6, dtype=np.float32).reshape(2, 3),
        'point': Point(1.5, 2.5),
        'cmap': cmap,
        'when': datetime.datetime(2026, 1, 2, 3, 4, 5),
        'nested': {'a': [1, 2], 'b': 'text'},
        (1, 2): 'tuple-key',
    }

    tf = tmpdir.join("config.cfg")
    configfile.writeConfigFile(data, tf)
    config = configfile.readConfigFile(tf)

    assert all(config['arr1d'] == data['arr1d'])
    assert config['arr2d'].dtype == np.float32
    assert all((config['arr2d'] == data['arr2d']).ravel())
    assert config['point'] == data['point']
    # ColorMap's repr round-trips through ColorMap(pos, color); compare
    # against the same reconstruction rather than the original object.
    assert config['cmap'] == ColorMap(cmap.pos, cmap.color)
    assert config['when'] == data['when']
    assert config['nested'] == data['nested']
    assert config[(1, 2)] == 'tuple-key'


def test_deeply_nested_expression_is_contained(tmpdir):
    """
    Test that an extremely deeply nested value fails with ParseError
    instead of escaping as a RecursionError or crashing the load, while
    moderate nesting still parses.
    """
    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write("x: " + "[" * 2000 + "1" + "]" * 2000 + "\n")

    with pytest.raises(configfile.ParseError):
        configfile.readConfigFile(tf)

    # Moderate nesting still parses: only pathological depth is refused.
    tf2 = tmpdir.join("shallow.cfg")
    with open(tf2, 'w') as f:
        f.write("x: " + "[" * 50 + "1" + "]" * 50 + "\n")
    config = configfile.readConfigFile(tf2)
    value = config['x']
    for _ in range(50):
        value = value[0]
    assert value == 1
