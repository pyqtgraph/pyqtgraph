import platform

from unittest.mock import MagicMock

import numpy as np

import pyqtgraph as pg
import pyqtgraph.parametertree as pt
from pyqtgraph.functions import eq
from pyqtgraph.parametertree.parameterTypes import ChecklistParameterItem
from pyqtgraph.Qt import QtCore, QtGui

app = pg.mkQApp()

def _getWidget(param):
    return list(param.items.keys())[0].widget


def test_opts():
    paramSpec = [
        dict(name='bool', type='bool', readonly=True),
        dict(name='color', type='color', readonly=True),
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
    if hasattr(QtCore, 'QString'):
        all_objs['qstring'] = QtCore.QString('xxxµ')

    # FIXME: str parameter fails with brush and pen
    if platform.python_implementation() == "PyPy":
        [all_objs.pop(key) for key in ['brush', 'pen']]

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
            param.setValue() is teasted with each value from *objs*, we expect
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
            raise Exception("Setting %s parameter value to %r raised %r." % (param, v, exc))
        
        raise Exception("Setting %s parameter value to %r should have raised an exception." % (param, v))
        
        
def test_limits_enforcement():
    p = pt.Parameter.create(name='params', type='group', children=[
        dict(name='float', type='float', limits=[0, 1]),
        dict(name='int', type='int', bounds=[0, 1]),
        dict(name='list', type='list', limits=['x', 'y']),
        dict(name='dict', type='list', limits={'x': 1, 'y': 2}),
    ])
    t = pt.ParameterTree()
    t.setParameters(p)
    for k, vin, vout in [('float', -1, 0),
                         ('float',  2, 1),
                         ('int',   -1, 0),
                         ('int',    2, 1),
                         ('list',   'w', 'x'),
                         ('dict',   'w', 1)]:
        p[k] = vin
        assert p[k] == vout


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
