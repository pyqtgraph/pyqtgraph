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

      - displaying a custom prefix (unit title)
      - displaying units with si prefix
      - built-in averaging
    """

    def __init__(self, parent=None, prefix='', suffix='', siPrefix=False, decimals=3,
                 averageTime=0., formatStr='', fancyMinus=True):
        """
        ============== ==================================================================================
        **Arguments:**
        prefix         (str) The prefix to place before the value
        suffix         (str) The suffix to place after the value
        siPrefix       (bool) Whether to add an SI prefix to the units and display a scaled value
        decimals       (int) Number of decimal values to display. Default is 3.
        averageTime    (float) The length of time in seconds to average values. If this value
                       is 0, then no averaging is performed. As this value increases
                       the display value will appear to change more slowly and smoothly.
        formatStr      (str) Formatting string used to generate the text shown. Formatting is
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

                       The default value is '{prefix}{prefixGap}{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}'
        fancyMinus     (bool) Whether to replace '-' with '−'. Default is True.
        ============== ==================================================================================
        """
        QtWidgets.QLabel.__init__(self, parent)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
                                     | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.values = []
        self.opts = {
            'averageTime': max(averageTime, 0.),
            'prefix': prefix or '',
            'suffix': suffix or '',
            'siPrefix': bool(siPrefix),
            'decimals': max(int(decimals), 0),
            'fancyMinus': fancyMinus,
            'format': formatStr or '{prefix}{prefixGap}{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}',
        }

    @property
    def value(self):
        return self.averageValue()

    @value.setter
    def value(self, new_value):
        self.setValue(float(new_value))

    @property
    def prefix(self):
        """ String to be prepended to the value """
        return self.opts['prefix']

    @prefix.setter
    def prefix(self, new_value):
        """ String to be prepended to the value """
        self.opts['prefix'] = new_value
        self.update()

    @property
    def title(self):
        """ String to be prepended to the value """
        return self.opts['prefix']

    @title.setter
    def title(self, new_value):
        """ String to be prepended to the value """
        self.opts['prefix'] = new_value
        self.update()

    @property
    def suffix(self):
        """ Suffix (units) to display after the numerical value """
        return self.opts['suffix']

    @suffix.setter
    def suffix(self, new_value):
        """ Suffix (units) to display after the numerical value """
        self.opts['suffix'] = new_value
        self.update()

    @property
    def siPrefix(self):
        return self.opts['siPrefix']

    @siPrefix.setter
    def siPrefix(self, new_value):
        self.opts['siPrefix'] = new_value
        self.update()

    @property
    def si_prefix(self):
        """
        If True, then an SI prefix is automatically prepended
        to the units and the value is scaled accordingly. For example,
        if value=0.003 and suffix='V', then the ValueLabel will display
        "300 mV" (but ValueLabel.value will still be 0.003). In case
        the value represents a dimensionless quantity that might span many
        orders of magnitude, such as a Reynolds number, an SI
        prefix is allowed with no suffix.
        :return: whether SI suffix is prepended to the unit
        """
        return self.opts['siPrefix']

    @si_prefix.setter
    def si_prefix(self, new_value):
        """
        If True, then an SI prefix is automatically prepended
        to the units and the value is scaled accordingly. For example,
        if value=0.003 and suffix='V', then the ValueLabel will display
        "300 mV" (but ValueLabel.value will still be 0.003). In case
        the value represents a dimensionless quantity that might span many
        orders of magnitude, such as a Reynolds number, an SI
        prefix is allowed with no suffix.
        :param new_value (bool) whether SI suffix is prepended to the unit
        """
        self.opts['siPrefix'] = new_value
        self.update()

    @property
    def decimals(self):
        """ Number of decimal values to display """
        return self.opts['decimals']

    @decimals.setter
    def decimals(self, new_value):
        """ Number of decimal values to display """
        self.opts['decimals'] = new_value
        self.update()

    @property
    def averageTime(self):
        """
        The length of time in seconds to average values. If this value
        is 0, then no averaging is performed. As this value increases
        the display value will appear to change more slowly and smoothly.
        :return: the length of time in seconds to average values
        """
        return self.opts['averageTime']

    @averageTime.setter
    def averageTime(self, new_value):
        """
        The length of time in seconds to average values. If this value
        is 0, then no averaging is performed. As this value increases
        the display value will appear to change more slowly and smoothly.
        :param new_value (float) the length of time in seconds to average values
        """
        self.opts['averageTime'] = new_value
        self.update()

    @property
    def average_time(self):
        """
        The length of time in seconds to average values. If this value
        is 0, then no averaging is performed. As this value increases
        the display value will appear to change more slowly and smoothly.
        :return: the length of time in seconds to average values
        """
        return self.opts['averageTime']

    @average_time.setter
    def average_time(self, new_value):
        """
        The length of time in seconds to average values. If this value
        is 0, then no averaging is performed. As this value increases
        the display value will appear to change more slowly and smoothly.
        :param new_value (float) the length of time in seconds to average values
        """
        self.opts['averageTime'] = new_value
        self.update()

    @property
    def format(self):
        """
        Formatting string used to generate the text shown. Formatting is
        done with ``str.format()`` and makes use of several arguments:

          * *prefix* - the prefix string
          * *prefixGap* - a single space if a prefix is present, or an empty
            string otherwise
          * *value* - the unscaled averaged value of the label
          * *scaledValue* - the scaled value to use when an SI prefix is present
          * *exp* and *mantissa* - the numbers so that value == mantissa * (10 ** exp)
          * *superscriptExp* - *exp* displayed with superscript symbols
          * *suffixGap* - a single space if a suffix is present, or an empty
            string otherwise.
          * *siPrefix* - the SI prefix string (if any), or an empty string if
            this feature has been disabled
          * *suffix* - the suffix string

        :return: the formatting string used to generate the text shown
        """
        return self.opts['format']

    @format.setter
    def format(self, new_value):
        """
        Formatting string used to generate the text shown. Formatting is
        done with ``str.format()`` and makes use of several arguments:

          * *prefix* - the prefix string
          * *prefixGap* - a single space if a prefix is present, or an empty
            string otherwise
          * *value* - the unscaled averaged value of the label
          * *scaledValue* - the scaled value to use when an SI prefix is present
          * *exp* and *mantissa* - the numbers so that value == mantissa * (10 ** exp)
          * *superscriptExp* - *exp* displayed with superscript symbols
          * *suffixGap* - a single space if a suffix is present, or an empty
            string otherwise.
          * *siPrefix* - the SI prefix string (if any), or an empty string if
            this feature has been disabled
          * *suffix* - the suffix string

        :param new_value (str) the formatting string used to generate the text shown
        """
        self.opts['format'] = new_value
        self.update()

    @property
    def formatStr(self):
        """
        Formatting string used to generate the text shown. Formatting is
        done with ``str.format()`` and makes use of several arguments:

          * *prefix* - the prefix string
          * *prefixGap* - a single space if a prefix is present, or an empty
            string otherwise
          * *value* - the unscaled averaged value of the label
          * *scaledValue* - the scaled value to use when an SI prefix is present
          * *exp* and *mantissa* - the numbers so that value == mantissa * (10 ** exp)
          * *superscriptExp* - *exp* displayed with superscript symbols
          * *suffixGap* - a single space if a suffix is present, or an empty
            string otherwise.
          * *siPrefix* - the SI prefix string (if any), or an empty string if
            this feature has been disabled
          * *suffix* - the suffix string

        :return: the formatting string used to generate the text shown
        """
        return self.opts['format']

    @formatStr.setter
    def formatStr(self, new_value):
        """
        Formatting string used to generate the text shown. Formatting is
        done with ``str.format()`` and makes use of several arguments:

          * *prefix* - the prefix string
          * *prefixGap* - a single space if a prefix is present, or an empty
            string otherwise
          * *value* - the unscaled averaged value of the label
          * *scaledValue* - the scaled value to use when an SI prefix is present
          * *exp* and *mantissa* - the numbers so that value == mantissa * (10 ** exp)
          * *superscriptExp* - *exp* displayed with superscript symbols
          * *suffixGap* - a single space if a suffix is present, or an empty
            string otherwise.
          * *siPrefix* - the SI prefix string (if any), or an empty string if
            this feature has been disabled
          * *suffix* - the suffix string

        :param new_value (str) the formatting string used to generate the text shown
        """
        self.opts['format'] = new_value
        self.update()

    @property
    def fancyMinus(self):
        """ Whether to replace '-' with '−' in the label shown """
        return self.opts['fancyMinus']

    @fancyMinus.setter
    def fancyMinus(self, new_value):
        """ Whether to replace '-' with '−' in the label shown """
        self.opts['fancyMinus'] = new_value

    @property
    def fancy_minus(self):
        """ Whether to replace '-' with '−' in the label shown """
        return self.opts['fancyMinus']

    @fancy_minus.setter
    def fancy_minus(self, new_value):
        """ Whether to replace '-' with '−' in the label shown """
        self.opts['fancyMinus'] = new_value

    def setOpts(self, **opts):
        """Set options affecting the behavior of the ValueLabel.

        =================== ========================================================================
        **Arguments:**
        prefix              (str) String to be prepended to the value. Default is an empty string.
        suffix              (str) Suffix (units) to display after the numerical value. By default,
                            the suffix is an empty str.
        siPrefix            (bool) If True, then an SI prefix is automatically prepended
                            to the units and the value is scaled accordingly. For example,
                            if value=0.003 and suffix='V', then the ValueLabel will display
                            "300 mV" (but ValueLabel.value will still be 0.003). In case
                            the value represents a dimensionless quantity that might span many
                            orders of magnitude, such as a Reynolds number, an SI
                            prefix is allowed with no suffix. Default is False.
        decimals            (int) Number of decimal values to display. Default is 3.
        averageTime         (float) The length of time in seconds to average values. If this value
                            is 0, then no averaging is performed. As this value increases
                            the display value will appear to change more slowly and smoothly.
        format or formatStr (str) Formatting string used to generate the text shown. Formatting is
                            done with ``str.format()`` and makes use of several arguments:
                              * *prefix* - the prefix string
                              * *prefixGap* - a single space if a prefix is present, or an empty
                                string otherwise
                              * *value* - the unscaled averaged value of the label
                              * *scaledValue* - the scaled value to use when an SI prefix is present
                              * *exp* and *mantissa* - the numbers so that value == mantissa * (10 ** exp)
                              * *superscriptExp* - *exp* displayed with superscript symbols
                              * *suffixGap* - a single space if a suffix is present, or an empty
                                string otherwise
                              * *siPrefix* - the SI prefix string (if any), or an empty string if
                                this feature has been disabled
                              * *suffix* - the suffix string
        fancyMinus          (bool) Whether to replace '-' with '−'. Default is True.
        =================== ========================================================================
        """

        for k, v in opts.items():
            if k in self.opts:
                self.opts[k] = v
            else:
                raise TypeError("Invalid keyword argument '%s'." % k)
        if opts:
            self.update()

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
                 'exp': exp, 'mantissa': man}
        if 'superscriptExp' in self.formatStr:
            parts['superscriptExp'] = superscript_number(exp)
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
