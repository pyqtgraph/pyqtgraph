from unittest.mock import MagicMock

import numpy as np

import pyqtgraph as pg
import pyqtgraph.parametertree as pt
from pyqtgraph.functions import eq
from pyqtgraph.parametertree.utils import compare_parameters
from pyqtgraph.parametertree.parameterTypes import ChecklistParameterItem
from pyqtgraph.Qt import QtCore, QtGui

import pytest

app = pg.mkQApp()


def _getWidget(param):
    return list(param.items.keys())[0].widget


def test_typeless_param():
    p = pt.Parameter.create(name='test', type=None, value=set())
    p.setValue(range(4))


def test_opts():
    paramSpec = [
        dict(name='bool', type='bool', readonly=True),
        dict(name='color', type='color', readonly=True),
        dict(name='float', type='float', limits=None),
    ]

    param = pt.Parameter.create(name='params', type='group', children=paramSpec)
    tree = pt.ParameterTree()
    tree.setParameters(param)

    assert _getWidget(param.param('bool')).isEnabled() is False
    assert _getWidget(param.param('bool')).isEnabled() is False


def test_types():
    paramSpec = [
        dict(name='float', type='float'),
        dict(name='int', type='int'),
        dict(name='str', type='str'),
        dict(name='list', type='list', limits=['x','y','z']),
        dict(name='dict', type='list', limits={'x':1, 'y':3, 'z':7}),
        dict(name='bool', type='bool'),
        dict(name='color', type='color'),
    ]
    
    param = pt.Parameter.create(name='params', type='group', children=paramSpec)
    tree = pt.ParameterTree()
    tree.setParameters(param)

    all_objs = {
        'int0': 0, 'int':7, 'float': -0.35, 'bigfloat': 1e129, 'npfloat': np.float64(5), 
        'npint': np.int64(5),'npinf': np.inf, 'npnan': np.nan, 'bool': True, 
        'complex': 5+3j, 'str': '#xxx', 'unicode': 'µ', 
        'list': [1,2,3], 'dict': {'1': 2}, 'color': pg.mkColor('k'), 
        'brush': pg.mkBrush('k'), 'pen': pg.mkPen('k'), 'none': None
    }

    # float
    types = ['int0', 'int', 'float', 'bigfloat', 'npfloat', 'npint', 'npinf', 'npnan', 'bool']
    check_param_types(param.child('float'), float, float, 0.0, all_objs, types)

    # int
    types = ['int0', 'int', 'float', 'bigfloat', 'npfloat', 'npint', 'bool']
    check_param_types(param.child('int'), int, int, 0, all_objs, types)
    
    # str  (should be able to make a string out of any type)
    types = all_objs.keys()
    check_param_types(param.child('str'), str, str, '', all_objs, types)
    
    # bool  (should be able to make a boolean out of any type?)
    types = all_objs.keys()
    check_param_types(param.child('bool'), bool, bool, False, all_objs, types)

    # color
    types = ['color', 'int0', 'int', 'float', 'npfloat', 'npint', 'list']
    init = QtGui.QColor(128, 128, 128, 255)
    check_param_types(param.child('color'), QtGui.QColor, pg.mkColor, init, all_objs, types)

    
def check_param_types(param, types, map_func, init, objs, keys):
    """Check that parameter setValue() accepts or rejects the correct types and
    that value() returns the correct type.
    
    Parameters
    ----------
        param : Parameter instance
        types : type or tuple of types
            The allowed types for this parameter to return from value().
        map_func : function
            Converts an input value to the expected output value.
        init : object
            The expected initial value of the parameter
        objs : dict
            Contains a variety of objects that will be tested as arguments to
            param.setValue().
        keys : list
            The list of keys indicating the valid objects in *objs*. When
            param.setValue() is tested with each value from *objs*, we expect
            an exception to be raised if the associated key is not in *keys*.
    """
    val = param.value()
    if not isinstance(types, tuple):
        types = (types,)
    assert val == init and type(val) in types

    # test valid input types
    good_inputs = [objs[k] for k in keys if k in objs]
    good_outputs = map(map_func, good_inputs)
    for x,y in zip(good_inputs, good_outputs):
        param.setValue(x)
        val = param.value()
        if not (eq(val, y) and type(val) in types):
            raise Exception("Setting parameter %s with value %r should have resulted in %r (types: %r), "
                "but resulted in %r (type: %r) instead." % (param, x, y, types, val, type(val)))

    # test invalid input types
    for k,v in objs.items():
        if k in keys:
            continue
        try:
            param.setValue(v)
        except (TypeError, ValueError, OverflowError):
            continue
        except Exception as exc:
            raise Exception(
                "Setting %s parameter value to %r raised %r." % (param, v, exc)
            ) from exc

        raise Exception("Setting %s parameter value to %r should have raised an exception." % (param, v))
        

@pytest.mark.parametrize("k,v_in,v_out",[
    ('float', -1, 0),
    ('float',  2, 1),
    ('int',   -1, 0),
    ('int',    2, 1),
    ('list', 'w', 'x'),
    ('dict', 'w', 1)
])
def test_limits_enforcement(k, v_in, v_out):
    p = pt.Parameter.create(name='params', type='group', children=[
        dict(name='float', type='float', limits=[0, 1]),
        dict(name='int', type='int', bounds=[0, 1]),
        dict(name='list', type='list', limits=['x', 'y']),
        dict(name='dict', type='list', limits={'x': 1, 'y': 2}),
    ])
    t = pt.ParameterTree()
    t.setParameters(p)
    p[k] = v_in
    assert p[k] == v_out


def test_set_to_default_updates_reset_button():
    p = pt.Parameter.create(name='int', type='int', value=1, default=2)
    tree = pt.ParameterTree()
    tree.addParameters(p)

    item = next(iter(p.items))
    assert item.defaultBtn.isEnabled() is True

    p.setToDefault()
    assert item.defaultBtn.isEnabled() is False


def test_data_race():
    # Ensure widgets don't override user setting of param values whether
    # they connect the signal before or after it's added to a tree
    p = pt.Parameter.create(name='int', type='int', value=0)
    t = pt.ParameterTree()

    def override():
        p.setValue(1)

    p.sigValueChanged.connect(override)
    t.setParameters(p)
    pi = next(iter(p.items))
    assert pi.param is p
    pi.widget.setValue(2)
    assert p.value() == pi.widget.value() == 1
    p.sigValueChanged.disconnect(override)
    p.sigValueChanged.connect(override)
    pi.widget.setValue(2)
    assert p.value() == pi.widget.value() == 1


def test_checklist_show_hide():
    p = pt.Parameter.create(name='checklist', type='checklist', limits=["a", "b", "c"])
    pi = ChecklistParameterItem(p, 0)
    pi.setHidden = MagicMock()
    p.hide()
    pi.setHidden.assert_called_with(True)
    assert not p.opts["visible"]
    p.show()
    pi.setHidden.assert_called_with(False)
    assert p.opts["visible"]

@pytest.mark.parametrize("limits,value",[
    ([1, 2, 3], [1, 2, 3]),
    ([1, 2, 3],  []),
    (['a', 'b', 'c'], ['a', 'b', 'c']),
    (['a', 'b', 'c'], []),
])
def test_checklist_check_and_clear_all(limits, value):
    p = pt.Parameter.create(name='checklist', type='checklist', limits=limits, value=value)
    pi = ChecklistParameterItem(p, 0)

    clearButton = pi.metaBtns['Clear']
    selectButton = pi.metaBtns['Select']
    
    # ensure only the specified ones are selected by default
    assert pi.param.value() == value

    # make all are selected after selecting all
    selectButton.clicked.emit()
    assert pi.param.value() == limits

    # make sure they all get cleared when hitting clear all
    clearButton.clicked.emit()
    assert pi.param.value() == []

    # make sure all are selected again
    selectButton.clicked.emit()
    assert pi.param.value() == limits


def test_pen_settings():
    # Option from constructor
    p = pt.Parameter.create(name='test', type='pen', width=5, additionalname='test')
    assert p.pen.width() == 5
    # Opts from dynamic update
    p.setOpts(width=3)
    assert p.pen.width() == 3
    # Opts from changing child
    p["width"] = 10
    assert p.pen.width() == 10


def test_recreate_from_savestate():
    from pyqtgraph.examples import _buildParamTypes
    created = _buildParamTypes.makeAllParamTypes()
    state = created.saveState()
    created2 = pt.Parameter.create(**state)
    assert pg.eq(state, created2.saveState())
    assert compare_parameters(created, created2)


# ---------------------------------------------------------------------------
# JSON serialization tests
#
# The fixture mirrors test_recreate_from_savestate: makeAllParamTypes() builds
# the same parameter tree used in examples/parametertree.py, so JSON behaviour
# is validated against the canonical example rather than ad-hoc specs.
# Targeted tests (pen/color/limits) exercise specific encoder edge cases that
# the full-tree test would catch but not diagnose clearly on failure.
# ---------------------------------------------------------------------------

import json as _json

import pytest

from pyqtgraph.parametertree.iojson import (
    parameter_from_json,
    parameter_from_json_file,
    parameter_restore_from_json,
    parameter_restore_from_json_file,
    parameter_to_json,
    parameter_to_json_file,
)


@pytest.fixture
def example_params():
    """Full parameter tree from examples/parametertree.py."""
    from pyqtgraph.examples import _buildParamTypes
    return _buildParamTypes.makeAllParamTypes()


# --- round-trip correctness --------------------------------------------------

def test_json_round_trip(example_params):
    """Full example tree survives a JSON string round-trip unchanged."""
    original_state = example_params.saveState()
    restored = parameter_from_json(parameter_to_json(example_params))
    assert pg.eq(original_state, restored.saveState())
    assert compare_parameters(example_params, restored)


def test_json_file_round_trip(example_params, tmp_path):
    """Full example tree survives a JSON file round-trip unchanged."""
    original_state = example_params.saveState()
    dest = tmp_path / 'params.json'
    parameter_to_json_file(example_params, dest)
    assert dest.exists()
    restored = parameter_from_json_file(dest)
    assert pg.eq(original_state, restored.saveState())
    assert compare_parameters(example_params, restored)


# --- encoder edge cases ------------------------------------------------------

def test_json_tuple_preservation():
    """Tuples survive as tuples, not lists.

    PenParameter.mkPen() checks isinstance(v, tuple) to detect its serialized
    form, so silently converting tuples to lists breaks pen restoration.
    """
    pen = pt.Parameter.create(name='pen', type='pen')
    original_state = pen.saveState()
    assert isinstance(original_state['value'], tuple), \
        "PenParameter.saveState() must return a tuple value"

    restored = parameter_from_json(parameter_to_json(pen))
    restored_state = restored.saveState()
    assert isinstance(restored_state['value'], tuple), \
        "Tuple value must be preserved through JSON round-trip"
    assert pg.eq(original_state, restored_state)
    assert compare_parameters(pen, restored)


def test_json_color_parameter():
    """Color parameter (QColor → RGBA tuple via saveState) round-trips correctly."""
    p = pt.Parameter.create(name='c', type='color', value='#ff8800')
    restored = parameter_from_json(parameter_to_json(p))
    assert pg.eq(p.saveState(), restored.saveState())
    assert compare_parameters(p, restored)


def test_json_limits_tuple():
    """Tuple-valued opts such as ``limits`` survive as tuples."""
    p = pt.Parameter.create(name='root', type='group', children=[
        dict(name='x', type='float', value=1.0, limits=(0.0, 10.0)),
    ])
    restored = parameter_from_json(parameter_to_json(p))
    assert isinstance(restored.saveState()['children']['x'].get('limits'), tuple)
    assert compare_parameters(p, restored)


# --- filter='user' and restore path ------------------------------------------

def test_json_filter_user(example_params):
    """filter='user' produces a compact values-only file; restore updates the tree."""
    float_child = example_params.child('Sample Float').child('widget')
    float_child.setValue(3.14)

    user_json = parameter_to_json(example_params, filter='user')
    user_state = _json.loads(user_json)

    # User-filtered state carries no 'type', 'limits', etc. — only values
    assert 'type' not in user_state

    # Restore into a fresh copy and verify the changed value came through
    fresh = _buildParamTypes_makeAllParamTypes()
    parameter_restore_from_json(fresh, user_json)
    assert fresh.child('Sample Float').child('widget').value() == pytest.approx(3.14)


def test_json_restore_preserves_signals(example_params):
    """restoreState path fires signals but keeps existing connections alive."""
    float_child = example_params.child('Sample Float').child('widget')
    float_child.setValue(1.0)

    received = []
    float_child.sigValueChanged.connect(lambda p, v: received.append(v))

    # Capture state at 1.0, mutate to 9.0, then restore — signal must fire
    json_str = parameter_to_json(example_params)
    float_child.setValue(9.0)
    parameter_restore_from_json(example_params, json_str)

    assert float_child.value() == pytest.approx(1.0)
    assert len(received) >= 1, "sigValueChanged must fire through restoreState"


def test_json_restore_from_file(example_params, tmp_path):
    """parameter_restore_from_json_file updates an existing tree from a file."""
    float_child = example_params.child('Sample Float').child('widget')
    float_child.setValue(2.71)

    parameter_to_json_file(example_params, tmp_path / 'settings', filter='user')

    fresh = _buildParamTypes_makeAllParamTypes()
    parameter_restore_from_json_file(fresh, tmp_path / 'settings.json')
    assert fresh.child('Sample Float').child('widget').value() == pytest.approx(2.71)


# --- overwrite guard ---------------------------------------------------------

def test_json_file_no_overwrite(example_params, tmp_path):
    """parameter_to_json_file raises FileExistsError when overwrite=False."""
    dest = tmp_path / 'params.json'
    parameter_to_json_file(example_params, dest)
    with pytest.raises(FileExistsError):
        parameter_to_json_file(example_params, dest, overwrite=False)


# --- public API surface -------------------------------------------------------

def test_json_accessible_from_package():
    """All six functions are importable directly from pyqtgraph.parametertree."""
    import pyqtgraph.parametertree as ptt
    for name in (
        'parameter_to_json', 'parameter_from_json', 'parameter_restore_from_json',
        'parameter_to_json_file', 'parameter_from_json_file',
        'parameter_restore_from_json_file',
    ):
        assert callable(getattr(ptt, name))


# helper — avoids repeating the import in every test body
def _buildParamTypes_makeAllParamTypes():
    from pyqtgraph.examples import _buildParamTypes
    return _buildParamTypes.makeAllParamTypes()
