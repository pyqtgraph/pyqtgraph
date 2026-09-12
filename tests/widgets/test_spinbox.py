import math
import re

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
    (123456.789, '123457', dict(int=False)),#No group separator expected
])
def test_SpinBox_formatting(value, expected_text, opts):
    if 'e' in expected_text:
        expect_failure_on_buggy_qt()
    
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
    (123456.789, '123457', dict(int=False)),#No group separator expected
])
def test_SpinBox_formatting_with_comma_decimal_separator(value, expected_text, opts):
    if 'e' in expected_text:
        expect_failure_on_buggy_qt()
           
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


def test_SpinBox_reformats_unchanged_integer_value():
    sb = pg.SpinBox(int=True)
    sb.setValue(10)

    sb.lineEdit().setText('2.5')
    sb.editingFinishedEvent()
    assert sb.value() == 2
    assert sb.text() == '2'

    sb.lineEdit().setText('2.1')
    sb.editingFinishedEvent()
    assert sb.value() == 2
    assert sb.text() == '2'


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


# Locale-formatted number input. Inputs come from a SpinBox's own display or
# from the spec; expected values come from the typed number, a C-locale
# SpinBox or float(), never from Qt's locale strings. A test is skipped when
# this Qt build's data for its locale differs from what the case needs.

DIRECTION_MARKS = ('\u200e', '\u200f', '\u061c')


def _localeOrSkip(name, reason, check):
    """Return the QLocale *name*, or skip when this Qt build's data fails *check*."""
    locale = pg.QtCore.QLocale(name)
    if locale.name() != name or not check(locale):
        pytest.skip(f"{name}: {reason} in this Qt build")
    return locale


def _shown(value, locale, **opts):
    """Return the text a SpinBox with *opts* in *locale* displays for *value*."""
    return pg.SpinBox(value=value, locale=locale, **opts).text()


def _enterText(sb, text):
    """Put *text* in the editor, finish editing and return the committed value."""
    sb.lineEdit().setText(text)
    sb.editingFinishedEvent()
    return sb.value()


def _assertRejected(sb, text):
    before = sb.value()
    _enterText(sb, text)
    assert sb.value() == before
    assert sb.lineEdit().text() == text
    assert sb.validate(text, 0)[0] == pg.QtGui.QValidator.State.Intermediate


def _sameValue(value, expected):
    if math.isnan(expected):
        return math.isnan(value)
    return value == expected


def _otherExponentCase(locale, text):
    exponent = locale.exponential()
    if exponent.lower() in text:
        return text.replace(exponent.lower(), exponent.upper())
    return text.replace(exponent.upper(), exponent.lower())


def _exponentPlusSignHasMark(locale):
    plus = locale.positiveSign()
    return any(mark in plus for mark in DIRECTION_MARKS) and plus in _shown(1500000, locale)


def _commaAndDisplayedMinusSign(locale):
    return locale.decimalPoint() == ',' and _shown(-1.5, locale) == '\u2212' + _shown(1.5, locale)


def _arabicIndic(locale):
    return locale.decimalPoint() == '\u066b' and locale.zeroDigit() == '\u0660'


def _minusSignAndComma(locale):
    return locale.negativeSign() == '\u2212' and locale.decimalPoint() == ','


def _asciiDigitsAndPeriod(locale):
    return locale.zeroDigit() == '0' and locale.decimalPoint() == '.'


def _nonAsciiExponent(locale):
    return locale.exponential() not in ('e', 'E')


def _cyrillicExponent(locale):
    return locale.exponential().lower() == '\u0435'


SI = dict(suffix='V', siPrefix=True)


@pytest.mark.parametrize("localeName, reason, check, value, opts, edit", [
    # Decimal separator other than period or comma
    pytest.param('ar_EG', "decimal separator is '.' or ',' or digits are ASCII",
                 lambda loc: loc.decimalPoint() not in ('.', ',') and loc.zeroDigit() != '0',
                 1.5, {}, None, id='decimal-separator-ar_EG'),
    # Non-ASCII digits with a period separator
    pytest.param('bn_BD', "decimal separator is not '.' or digits are ASCII",
                 lambda loc: loc.decimalPoint() == '.' and not loc.zeroDigit().isascii(),
                 1.5, {}, None, id='non-ascii-digits-bn_BD'),
    # Minus signs other than a plain hyphen-minus
    pytest.param('sv_SE', "minus sign is not U+2212", lambda loc: loc.negativeSign() == '\u2212',
                 -1.5, {}, None, id='minus-sign-sv_SE'),
    pytest.param('he_IL', "minus sign has no U+200E mark", lambda loc: '\u200e' in loc.negativeSign(),
                 -1.5, {}, None, id='minus-sign-mark-he_IL'),
    pytest.param('ar_EG', "minus sign has no U+061C mark", lambda loc: '\u061c' in loc.negativeSign(),
                 -1.5, {}, None, id='minus-sign-mark-ar_EG'),
    # Direction mark in the exponent's plus sign
    pytest.param('he_IL', "no direction mark in the displayed exponent sign", _exponentPlusSignHasMark,
                 1500000, {}, None, id='exponent-plus-mark-he_IL'),
    pytest.param('ar_EG', "no direction mark in the displayed exponent sign", _exponentPlusSignHasMark,
                 1500000, {}, None, id='exponent-plus-mark-ar_EG'),
    # Exponent symbol other than e
    pytest.param('sv_SE', "exponent symbol is e", _nonAsciiExponent,
                 1500000, {}, None, id='exponent-symbol-sv_SE'),
    pytest.param('sv_SE', "exponent symbol is e", _nonAsciiExponent,
                 -0.0000015, {}, None, id='exponent-symbol-negative-sv_SE'),
    pytest.param('uk_UA', "exponent symbol is e", _nonAsciiExponent,
                 1500000, {}, None, id='exponent-symbol-uk_UA'),
    pytest.param('uk_UA', "exponent symbol is e", _nonAsciiExponent,
                 -0.0000015, {}, None, id='exponent-symbol-negative-uk_UA'),
    # Exponent symbol in either letter case
    pytest.param('uk_UA', "exponent symbol is ASCII or has no letter case",
                 lambda loc: not loc.exponential().isascii() and loc.exponential().lower() != loc.exponential().upper(),
                 1500000, {}, None, id='exponent-case-shown-uk_UA'),
    pytest.param('uk_UA', "exponent symbol is ASCII or has no letter case",
                 lambda loc: not loc.exponential().isascii() and loc.exponential().lower() != loc.exponential().upper(),
                 1500000, {}, _otherExponentCase, id='exponent-case-other-uk_UA'),
    # SI prefix and unit suffix
    pytest.param('sv_SE', "minus sign and decimal separator are ASCII",
                 lambda loc: not loc.negativeSign().isascii() or not loc.decimalPoint().isascii(),
                 0.0015, SI, None, id='si-prefix-sv_SE'),
    pytest.param('sv_SE', "minus sign and decimal separator are ASCII",
                 lambda loc: not loc.negativeSign().isascii() or not loc.decimalPoint().isascii(),
                 -0.0015, SI, None, id='si-prefix-negative-sv_SE'),
    pytest.param('ar_EG', "minus sign and decimal separator are ASCII",
                 lambda loc: not loc.negativeSign().isascii() or not loc.decimalPoint().isascii(),
                 0.0015, SI, None, id='si-prefix-ar_EG'),
    pytest.param('ar_EG', "minus sign and decimal separator are ASCII",
                 lambda loc: not loc.negativeSign().isascii() or not loc.decimalPoint().isascii(),
                 -0.0015, SI, None, id='si-prefix-negative-ar_EG'),
    # Negative infinity when non-finite values are allowed
    pytest.param('sv_SE', "minus sign is a hyphen-minus", lambda loc: loc.negativeSign() != '-',
                 float('-inf'), dict(finite=False), None, id='negative-infinity-sv_SE'),
    pytest.param('ar_EG', "minus sign is a hyphen-minus", lambda loc: loc.negativeSign() != '-',
                 float('-inf'), dict(finite=False), None, id='negative-infinity-ar_EG'),
    # Suffix containing the locale's exponent symbol
    pytest.param('uk_UA', "exponent symbol is not a letter of the suffix",
                 lambda loc: _cyrillicExponent(loc) and '\u0435' in 'метр',
                 1500000, dict(suffix='метр'), None, id='suffix-with-exponent-symbol-uk_UA'),
    # Prefix containing the locale's exponent symbol
    pytest.param('uk_UA', "exponent symbol is not Cyrillic e", _cyrillicExponent,
                 2.5, dict(prefix='Величина'), None, id='prefix-with-exponent-symbol-uk_UA'),
    # Suffix containing the locale's decimal separator
    pytest.param('de_DE', "decimal separator is not ','", lambda loc: loc.decimalPoint() == ',',
                 2.5, dict(suffix='V,eff'), None, id='suffix-with-decimal-separator-de_DE'),
])
def test_SpinBox_reads_its_own_locale_display(localeName, reason, check, value, opts, edit):
    locale = _localeOrSkip(localeName, reason, check)
    text = _shown(value, locale, **opts)
    if edit is not None:
        text = edit(locale, text)
    sb = pg.SpinBox(locale=locale, **opts)

    sb.lineEdit().setText(text)
    assert sb.validate(text, 0)[0] == pg.QtGui.QValidator.State.Acceptable, text
    sb.editingFinishedEvent()
    assert _sameValue(sb.value(), value), text


@pytest.mark.parametrize("localeName, reason, check, text, opts, expected", [
    # Int mode
    pytest.param('sv_SE', "decimal separator is not ',' or the displayed minus sign is not U+2212",
                 _commaAndDisplayedMinusSign, '−3', dict(int=True), -3, id='int-mode-sv_SE'),
    pytest.param('sv_SE', "decimal separator is not ',' or the displayed minus sign is not U+2212",
                 _commaAndDisplayedMinusSign, '−2,5', dict(int=True),
                 lambda: _enterText(pg.SpinBox(int=True, locale=pg.QtCore.QLocale.c()), '-2.5'),
                 id='int-mode-fraction-sv_SE'),
    # Locale digits with an ASCII period
    pytest.param('ar_EG', "decimal separator is not U+066B or digits are not Arabic-Indic", _arabicIndic,
                 '١.٥', {}, 1.5, id='locale-digits-ascii-period-ar_EG'),
    # Another script's digits in a locale with ASCII digits
    pytest.param('en_US', "digits are not ASCII or decimal separator is not '.'", _asciiDigitsAndPeriod,
                 '১.৫', {}, 1.5, id='other-script-digits-en_US'),
    # ASCII signs in a locale with a different minus sign
    pytest.param('sv_SE', "minus sign is not U+2212 or decimal separator is not ','", _minusSignAndComma,
                 '-1,5', {}, -1.5, id='ascii-minus-sv_SE'),
    pytest.param('sv_SE', "minus sign is not U+2212 or decimal separator is not ','", _minusSignAndComma,
                 '+1,5', {}, 1.5, id='ascii-plus-sv_SE'),
    # ASCII minus in a locale with direction marks
    pytest.param('he_IL', "minus sign has no direction mark or decimal separator is not '.'",
                 lambda loc: any(mark in loc.negativeSign() for mark in DIRECTION_MARKS) and loc.decimalPoint() == '.',
                 '-1.5', {}, -1.5, id='ascii-minus-with-marks-he_IL'),
    # ASCII non-finite values
    pytest.param('sv_SE', "minus sign is a hyphen-minus", lambda loc: loc.negativeSign() != '-',
                 '-inf', dict(finite=False), float('-inf'), id='ascii-negative-infinity-sv_SE'),
    pytest.param('sv_SE', "minus sign is a hyphen-minus", lambda loc: loc.negativeSign() != '-',
                 '+inf', dict(finite=False), float('inf'), id='ascii-positive-infinity-sv_SE'),
    pytest.param('sv_SE', "minus sign is a hyphen-minus", lambda loc: loc.negativeSign() != '-',
                 'nan', dict(finite=False), float('nan'), id='ascii-nan-sv_SE'),
    # Period accepted where the separator is neither period nor comma
    pytest.param('ar_EG', "decimal separator is '.' or ','", lambda loc: loc.decimalPoint() not in ('.', ','),
                 '2.5', {}, 2.5, id='period-accepted-ar_EG'),
    # ASCII exponent marker in a locale with a different exponent symbol
    pytest.param('sv_SE', "exponent symbol is e or decimal separator is not ','",
                 lambda loc: _nonAsciiExponent(loc) and loc.decimalPoint() == ',',
                 '1,5e6', {}, 1500000, id='ascii-exponent-sv_SE'),
])
def test_SpinBox_reads_typed_text_in_locale(localeName, reason, check, text, opts, expected):
    locale = _localeOrSkip(localeName, reason, check)
    if callable(expected):
        expected = expected()
    sb = pg.SpinBox(locale=locale, **opts)

    assert _sameValue(_enterText(sb, text), expected)


@pytest.mark.parametrize("localeName, reason, check, text", [
    # Period rejected alongside other locale symbols
    pytest.param('sv_SE', "decimal separator is not ',' or the displayed minus sign is not U+2212",
                 _commaAndDisplayedMinusSign, '−1.5', id='period-with-minus-sign-sv_SE'),
    # Period rejected in a comma-decimal locale
    pytest.param('de_DE', "decimal separator is not ','", lambda loc: loc.decimalPoint() == ',',
                 '1.5', id='period-de_DE'),
    # Comma rejected where the separator is neither period nor comma
    pytest.param('ar_EG', "decimal separator is '.' or ','", lambda loc: loc.decimalPoint() not in ('.', ','),
                 '2,5', id='comma-ar_EG'),
])
def test_SpinBox_rejects_text_in_locale(localeName, reason, check, text):
    locale = _localeOrSkip(localeName, reason, check)
    sb = pg.SpinBox(locale=locale)

    _assertRejected(sb, text)
    assert sb.value() == 0


@pytest.mark.parametrize("localeName, reason, check, makeText, regex, accepts", [
    # Arabic-Indic digits and decimal separator
    pytest.param('ar_EG', "decimal separator is not U+066B or digits are not Arabic-Indic", _arabicIndic,
                 lambda loc: _shown(1.5, loc), None, lambda text: text == '1.5', id='arabic-indic-ar_EG'),
    # Minus sign other than hyphen-minus
    pytest.param('sv_SE', "minus sign is not U+2212 or decimal separator is not ','", _minusSignAndComma,
                 lambda loc: _shown(-1.5, loc), None, lambda text: text == '-1.5', id='minus-sign-sv_SE'),
    # Exponent symbol other than e passed as ASCII
    pytest.param('sv_SE', "exponent symbol is e", _nonAsciiExponent,
                 lambda loc: _shown(1500000, loc), None,
                 lambda text: text.isascii() and float(text) == 1500000, id='exponent-symbol-sv_SE'),
    pytest.param('uk_UA', "exponent symbol is e", _nonAsciiExponent,
                 lambda loc: _shown(1500000, loc), None,
                 lambda text: text.isascii() and float(text) == 1500000, id='exponent-symbol-uk_UA'),
    # ASCII text in a comma-decimal locale
    pytest.param('de_DE', "decimal separator is not ','", lambda loc: loc.decimalPoint() == ',',
                 lambda loc: '-1,5e6', None, lambda text: text == '-1.5e6', id='ascii-comma-de_DE'),
    # ASCII exponent marker kept as typed
    pytest.param('en_US', "decimal separator is not '.' or exponent symbol is not e",
                 lambda loc: loc.decimalPoint() == '.' and loc.exponential() in ('e', 'E'),
                 lambda loc: '1.5E6', None, lambda text: text == '1.5E6', id='ascii-exponent-en_US'),
    # Another script's digits passed unchanged
    pytest.param('en_US', "digits are not ASCII or decimal separator is not '.'", _asciiDigitsAndPeriod,
                 lambda loc: '১.৫', None, lambda text: text == '১.৫', id='other-script-digits-en_US'),
    # Custom regex together with a custom evalFunc
    pytest.param('ar_EG', "decimal separator is not U+066B or digits are not Arabic-Indic", _arabicIndic,
                 lambda loc: _shown(1.5, loc), re.compile(r'(?P<number>[^\sA-Za-z]+)'),
                 lambda text: text == '1.5', id='custom-regex-ar_EG'),
])
def test_SpinBox_evalFunc_receives_ascii_number(localeName, reason, check, makeText, regex, accepts):
    locale = _localeOrSkip(localeName, reason, check)
    received = []

    def evalFunc(text):
        received.append(text)
        return 1.0

    if regex is None:
        sb = pg.SpinBox(locale=locale, evalFunc=evalFunc)
    else:
        # locale before regex: setLocale replaces a regex that was set before it
        sb = pg.SpinBox(locale=locale, regex=regex, evalFunc=evalFunc)
        assert sb.opts['regex'] is regex
    _enterText(sb, makeText(locale))

    assert received
    assert all(accepts(text) for text in received), received


def test_SpinBox_custom_regex_sees_text_as_entered():
    # Custom regex that requires the locale's minus sign
    locale = _localeOrSkip('sv_SE', "minus sign is not U+2212 or decimal separator is not ','", _minusSignAndComma)
    regex = re.compile('(?P<number>' + re.escape(locale.negativeSign()) + r'[^\sA-Za-z]+)')
    # locale before regex: setLocale replaces a regex that was set before it
    sb = pg.SpinBox(locale=locale, regex=regex)
    assert sb.opts['regex'] is regex

    assert _enterText(sb, _shown(-1.5, locale)) == -1.5



def expect_failure_on_buggy_qt():
    if (6, 0) <= pg.Qt.QtVersionInfo < (6, 9):
        pytest.xfail("A known bug in Qt 6.0.0 - 6.8.x causes scientific notation with 'g' format to use capital 'E' for the exponent.")
