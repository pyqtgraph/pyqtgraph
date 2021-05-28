from pyqtgraph.parametertree import Parameter


def test_parameter_hasdefault():
    opts = {'name': 'param', 'type': int, 'value': 1}

    # default unspecified
    p = Parameter(**opts)
    assert p.hasDefault()
    assert p.defaultValue() == opts["value"]

    p.setDefault(2)
    assert p.hasDefault()
    assert p.defaultValue() == 2

    # default specified
    p = Parameter(default=0, **opts)
    assert p.hasDefault()
    assert p.defaultValue() == 0

    # default specified as None
    p = Parameter(default=None, **opts)
    assert not p.hasDefault()


def test_unpack_parameter():
    # test that **unpacking correctly returns child name/value maps
    params = [
        dict(name='a', type='int', value=1),
        dict(name='b', type='str', value='2'),
        dict(name='c', type='float', value=3.0),
    ]
    p = Parameter.create(name='params', type='group', children=params)
    result = dict(**p)

    assert 'a' in result
    assert result['a'] == 1
    assert 'b' in result
    assert result['b'] == '2'
    assert 'c' in result
    assert result['c'] == 3.0
