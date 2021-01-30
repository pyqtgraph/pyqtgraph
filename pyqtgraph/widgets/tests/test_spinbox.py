# -*- coding: utf-8 -*-
import pytest
import pyqtgraph as pg

pg.mkQApp()


def test_SpinBox_defaults():
    sb = pg.SpinBox()
    assert sb.opts['decimals'] == 6
    assert sb.opts['int'] is False


@pytest.mark.parametrize("value,expected_text,opts", [
    (0, '0', dict(suffix='', siPrefix=False, dec=False, int=False)),
    (100, '100', dict()),
    (1000000, '1e+06', dict()),
    (1000, '1e+03', dict(decimals=2)),
    (1000000, '1000000 V', dict(int=True, suffix='V')),
    (12345678955, '12345678955', dict(int=True, decimals=100)),
    (1.45e-9, '1.45e-09 A', dict(int=False, decimals=6, suffix='A', siPrefix=False)),
    (1.45e-9, '1.45 nA', dict(int=False, decimals=6, suffix='A', siPrefix=True)),
    (1.45, '1.45 PSI', dict(int=False, decimals=6, suffix='PSI', siPrefix=True)),
    (1.45e-3, '1.45 mPSI', dict(int=False, decimals=6, suffix='PSI', siPrefix=True)),
    (-2500.3427, '$-2500.34', dict(int=False, format='${value:0.02f}')),
])
def test_SpinBox_formatting(value, expected_text, opts):
    sb = pg.SpinBox(**opts)
    sb.setValue(value)

    assert sb.value() == value
    assert pg.asUnicode(sb.text()) == expected_text


@pytest.mark.parametrize("suffix", ["", "V"])
def test_SpinBox_gui_set_value(suffix):
    sb = pg.SpinBox(suffix=suffix)
    sb.lineEdit().setText('0.1' + suffix)
    sb.editingFinishedEvent()
    assert sb.value() == 0.1
    if suffix != '':
        sb.lineEdit().setText('0.1 m' + suffix)
        sb.editingFinishedEvent()
        assert sb.value() == 0.1e-3
