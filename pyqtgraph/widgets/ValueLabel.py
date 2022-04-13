# -*- coding: utf-8 -*-

import math
import time

from .. import functions as fn
from ..Qt import QtCore, QtWidgets

__all__ = ['ValueLabel']


def superscript_number(number):
    ss_dict = {
        '0': '⁰',
        '1': '¹',
        '2': '²',
        '3': '³',
        '4': '⁴',
        '5': '⁵',
        '6': '⁶',
        '7': '⁷',
        '8': '⁸',
        '9': '⁹',
        '-': '⁻',
        '−': '⁻'
    }
    number = str(number)
    for d in ss_dict:
        number = number.replace(d, ss_dict[d])
    return number


class ValueLabel(QtWidgets.QLabel):
    """
    QLabel specifically for displaying numerical values.
    Extends QLabel adding some extra functionality:

      - displaying units with si prefix
      - built-in exponential averaging
    """

    def __init__(self, parent=None, prefix='', suffix='', siPrefix=False, decimals=3, averageTime=0., formatStr=None, fancyMinus=True):
        """
        ==============      ==================================================================================
        **Arguments:**
        suffix              (str) The suffix to place after the value
        siPrefix            (bool) Whether to add an SI prefix to the units and display a scaled value
        decimals            (int) Number of decimal values to display. Default is 6.
        averageTime         (float) The length of time in seconds to average values. If this value
                            is 0, then no averaging is performed. As this value increases
                            the display value will appear to change more slowly and smoothly.
        format              (str) Formatting string used to generate the text shown. Formatting is
                            done with ``str.format()`` and makes use of several arguments:

                              * *value* - the unscaled value
                              * *avgValue* - same as *scaledValue*
                              * *exp* and *mantissa* - the numbers so that value == mantissa * (10 ** exp)
                              * *superscriptExp* - *exp* displayed with superscript symbols
                              * *prefix* - the prefix string
                              * *prefixGap* - a single space if a prefix is present, or an empty
                                string otherwise
                              * *suffix* - the suffix string
                              * *scaledValue* - the scaled value to use when an SI prefix is present
                              * *siPrefix* - the SI prefix string (if any), or an empty string if
                                this feature has been disabled
                              * *suffixGap* - a single space if a suffix is present, or an empty
                                string otherwise.
        fancyMinus          (bool) Whether to replace '-' with '−'. Default is True.
        ==============      ==================================================================================
        """
        QtWidgets.QLabel.__init__(self, parent)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
                                     | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.values = []
        self.averageTime = averageTime  # no averaging by default
        self.prefix = prefix or ''
        self.suffix = suffix or ''
        self.siPrefix = siPrefix
        self.decimals = decimals
        self.fancyMinus = fancyMinus
        if formatStr is None:
            self.formatStr = '{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}'
        else:
            self.formatStr = formatStr

    def setValue(self, value):
        now = time.monotonic()
        self.values.append((now, value))
        cutoff = now - self.averageTime
        while len(self.values) > 0 and self.values[0][0] < cutoff:
            self.values.pop(0)
        self.update()

    def setFormatStr(self, text):
        self.formatStr = text
        self.update()

    def setAverageTime(self, t):
        self.averageTime = t

    def averageValue(self):
        if self.values:
            return sum(v[1] for v in self.values) / float(len(self.values))
        else:
            return math.nan

    def paintEvent(self, ev):
        self.setText(self.generateText())
        return super().paintEvent(ev)

    def generateText(self):
        if len(self.values) == 0:
            return ''
        val = self.averageValue()
        if math.isnan(val):
            return ''

        # format the string
        exp = int(math.floor(math.log10(abs(val)))) if val != 0.0 else 0
        man = val * math.pow(0.1, exp)
        parts = {'value': val, 'prefix': self.prefix, 'suffix': self.suffix, 'decimals': self.decimals,
                 'exp': exp, 'mantissa': man, 'superscriptExp': superscript_number(exp)}
        if self.siPrefix:
            # SI prefix was requested, so scale the value accordingly
            (s, p) = fn.siScale(val)
            parts.update({'siPrefix': p, 'scaledValue': s * val, 'avgValue': s * val})
        else:
            # no SI prefix/suffix requested; scale is 1
            parts.update({'siPrefix': '', 'scaledValue': val, 'avgValue': val})

        parts['prefixGap'] = ' ' if parts['prefix'] else ''
        parts['suffixGap'] = ' ' if (parts['suffix'] or parts['siPrefix']) else ''

        formatted_value = self.formatStr.format(**parts)
        if self.fancyMinus:
            formatted_value = formatted_value.replace('-', '−')
        return formatted_value
