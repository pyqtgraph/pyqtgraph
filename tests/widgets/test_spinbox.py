import pytest

import pyqtgraph as pg

pg.mkQApp()


def test_SpinBox_defaults():
    sb = pg.SpinBox()
    assert sb.opts['decimals'] == 6
    assert sb.opts['int'] is False

englishLocale = pg.QtCore.QLocale(pg.QtCore.QLocale.Language.English)
germanLocale = pg.QtCore.QLocale(pg.QtCore.QLocale.Language.German, pg.QtCore.QLocale.Country.Germany)

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
    (1000, '1 k', dict(siPrefix=True, suffix="")),
    (1.45e-9, 'i = 1.45e-09 A', dict(int=False, decimals=6, suffix='A', siPrefix=False, prefix='i =')),
    (0, '0 mV', dict(suffix='V', siPrefix=True, scaleAtZero=1e-3)),
    (0, '0 mV', dict(suffix='V', siPrefix=True, minStep=5e-6, scaleAtZero=1e-3)),
    (0, '0 mV', dict(suffix='V', siPrefix=True, step=1e-3)),
    (0, '0 mV', dict(suffix='V', dec=True, siPrefix=True, minStep=15e-3)),
])
def test_SpinBox_formatting(value, expected_text, opts):
    if 'e' in expected_text and compare_semantic_versions(pg.Qt.QtVersion, '6.9.0') < 0:
        pytest.xfail("A known bug in Qt < 6.9.0 causes scientific notation with 'g' format to use capital 'E' for the exponent.")
    
    sb = pg.SpinBox(**opts)
    sb.setLocale(englishLocale)
    sb.setValue(value)

    assert sb.value() == value
    assert sb.text() == expected_text



@pytest.mark.parametrize("value,expected_text,opts", [
    (0, '0', dict(suffix='', siPrefix=False, dec=False, int=False)),
    (100, '100', dict()),
    (1000000, '1e+06', dict()),
    (1000, '1e+03', dict(decimals=2)),
    (1000000, '1000000 V', dict(int=True, suffix='V')),
    (12345678955, '12345678955', dict(int=True, decimals=100)),
    (1.45e-9, '1,45e-09 A', dict(int=False, decimals=6, suffix='A', siPrefix=False)),
    (1.45e-9, '1,45 nA', dict(int=False, decimals=6, suffix='A', siPrefix=True)),
    (1.45, '1,45 PSI', dict(int=False, decimals=6, suffix='PSI', siPrefix=True)),
    (1.45e-3, '1,45 mPSI', dict(int=False, decimals=6, suffix='PSI', siPrefix=True)),
    (-2500.3427, '$-2500.34', dict(int=False, format='${value:0.02f}')),#format specifier provided, so decimal separator unaffected by locale
    (1000, '1 k', dict(siPrefix=True, suffix="")),
    (1.45e-9, 'i = 1,45e-09 A', dict(int=False, decimals=6, suffix='A', siPrefix=False, prefix='i =')),
    (0, '0 mV', dict(suffix='V', siPrefix=True, scaleAtZero=1e-3)),
    (0, '0 mV', dict(suffix='V', siPrefix=True, minStep=5e-6, scaleAtZero=1e-3)),
    (0, '0 mV', dict(suffix='V', siPrefix=True, step=1e-3)),
    (0, '0 mV', dict(suffix='V', dec=True, siPrefix=True, minStep=15e-3)),
])
def test_SpinBox_formatting_with_comma_decimal_separator(value, expected_text, opts):
    if 'e' in expected_text and compare_semantic_versions(pg.Qt.QtVersion, '6.9.0') < 0:
        pytest.xfail("A known bug in Qt < 6.9.0 causes scientific notation with 'g' format to use capital 'E' for the exponent.")
           
    sb = pg.SpinBox(**opts)
    sb.setLocale(germanLocale)
    sb.setValue(value)

    assert sb.value() == value
    assert sb.text() == expected_text

def test_evalFunc():
    sb = pg.SpinBox(evalFunc=lambda s: 100)

    sb.lineEdit().setText('3')
    sb.editingFinishedEvent()
    assert sb.value() == 100

    sb.lineEdit().setText('0')
    sb.editingFinishedEvent()
    assert sb.value() == 100


def spinBox_gui_set_value_test(expected, valueText, suffix, locale):
    sb = pg.SpinBox(suffix=suffix, locale=locale)

    sb.lineEdit().setText(f'{valueText}{suffix}')
    sb.editingFinishedEvent()
    assert sb.value() == expected

@pytest.mark.parametrize("expected,valueText,suffix", [(0.1, "0.1", ""), (0.1e-3, "0.1 m", "V"), (0, "0,325", "A")])
def test_SpinBox_gui_set_value_english(expected, valueText, suffix):
    spinBox_gui_set_value_test(expected, valueText, suffix, locale=englishLocale)

@pytest.mark.parametrize("expected,valueText,suffix", [(0.1, "0,1", ""), (0.1e-3, "0,1 m", "V"), (0, "0.325", "A")])
def test_SpinBox_gui_set_value_german(expected, valueText, suffix):
    spinBox_gui_set_value_test(expected, valueText, suffix, locale=germanLocale) 


def compare_semantic_versions(v1, v2):
    try:
        parts1 = [int(p) for p in v1.split('.')]
        parts2 = [int(p) for p in v2.split('.')]
        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1
        return 0
    except ValueError:
        return 0
