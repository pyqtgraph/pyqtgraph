import pyqtgraph as pg
pg.mkQApp()


def test_spinbox_formatting():
    sb = pg.SpinBox()
    assert sb.opts['decimals'] == 6
    assert sb.opts['int'] is False
    
    # table  of test conditions:
    # value, text, options
    conds = [
        (0, '0', dict(suffix='', siPrefix=False, dec=False, int=False)),
        (100, '100', dict()),
        (1000000, '1e+06', dict()),
        (1000, '1e+03', dict(decimals=2)),
        (1000000, '1e+06', dict(int=True, decimals=6)),
        (12345678955, '12345678955', dict(int=True, decimals=100)),
        (1.45e-9, '1.45e-09 A', dict(int=False, decimals=6, suffix='A', siPrefix=False)),
        (1.45e-9, '1.45 nA', dict(int=False, decimals=6, suffix='A', siPrefix=True)),
        (-2500.3427, '$-2500.34', dict(int=False, format='${value:0.02f}')),
    ]
    
    for (value, text, opts) in conds:
        sb.setOpts(**opts)
        sb.setValue(value)
        assert sb.value() == value
        assert pg.asUnicode(sb.text()) == text
