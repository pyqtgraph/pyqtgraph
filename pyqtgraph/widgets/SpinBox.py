# -*- coding: utf-8 -*-
from math import isnan, isinf
from decimal import Decimal as D  ## Use decimal to avoid accumulating floating-point errors
import decimal
import weakref
import re

from ..Qt import QtGui, QtCore
from ..python2_3 import asUnicode, basestring
from ..SignalProxy import SignalProxy
from .. import functions as fn


__all__ = ['SpinBox']


class SpinBox(QtGui.QAbstractSpinBox):
    """
    **Bases:** QtGui.QAbstractSpinBox
    
    Extension of QSpinBox widget for selection of a numerical value.     
    Adds many extra features:
    
    * SI prefix notation (eg, automatically display "300 mV" instead of "0.003 V")
    * Float values with linear and decimal stepping (1-9, 10-90, 100-900, etc.)
    * Option for unbounded values
    * Delayed signals (allows multiple rapid changes with only one change signal)
    * Customizable text formatting
    
    =============================  ==============================================
    **Signals:**
    valueChanged(value)            Same as QSpinBox; emitted every time the value 
                                   has changed.
    sigValueChanged(self)          Emitted when value has changed, but also combines
                                   multiple rapid changes into one signal (eg, 
                                   when rolling the mouse wheel).
    sigValueChanging(self, value)  Emitted immediately for all value changes.
    =============================  ==============================================
    """
    
    ## There's a PyQt bug that leaks a reference to the 
    ## QLineEdit returned from QAbstractSpinBox.lineEdit()
    ## This makes it possible to crash the entire program 
    ## by making accesses to the LineEdit after the spinBox has been deleted.
    ## I have no idea how to get around this..
    
    
    valueChanged = QtCore.Signal(object)     # (value)  for compatibility with QSpinBox
    sigValueChanged = QtCore.Signal(object)  # (self)
    sigValueChanging = QtCore.Signal(object, object)  # (self, value)  sent immediately; no delay.

    def __init__(self, parent=None, value=0.0, **kwargs):
        """
        ============== ========================================================================
        **Arguments:**
        parent         Sets the parent widget for this SpinBox (optional). Default is None.
        value          (float/int) initial value. Default is 0.0.
        ============== ========================================================================
        
        All keyword arguments are passed to :func:`setOpts`.
        """
        QtGui.QAbstractSpinBox.__init__(self, parent)
        self.lastValEmitted = None
        self.lastText = ''
        self.textValid = True  ## If false, we draw a red border
        self.setMinimumWidth(0)
        self._lastFontHeight = None
        
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.errorBox = ErrorBox(self.lineEdit())
        
        self.opts = {
            'bounds': [None, None],
            'wrapping': False,
           
            ## normal arithmetic step
            'step': D('0.01'),  ## if 'dec' is false, the spinBox steps by 'step' every time
                                ## if 'dec' is True, the step size is relative to the value
                                ## 'step' needs to be an integral divisor of ten, ie 'step'*n=10 for some integer value of n (but only if dec is True)
            'dec': False,   ## if true, does decimal stepping. ie from 1-10 it steps by 'step', from 10 to 100 it steps by 10*'step', etc. 
                            ## if true, minStep must be set in order to cross zero.
            
            'int': False, ## Set True to force value to be integer
            'finite': True,
            
            'suffix': '',
            'siPrefix': False,   ## Set to True to display numbers with SI prefix (ie, 100pA instead of 1e-10A)
            
            'delay': 0.3, ## delay sending wheel update signals for 300ms
            
            'delayUntilEditFinished': True,   ## do not send signals until text editing has finished
            
            'decimals': 6,
            
            'format': asUnicode("{scaledValue:.{decimals}g}{suffixGap}{siPrefix}{suffix}"),
            'regex': fn.FLOAT_REGEX,
            'evalFunc': D,
            
            'compactHeight': True,  # manually remove extra margin outside of text
        }
        
        self.decOpts = ['step', 'minStep']
        
        self.val = D(asUnicode(value))  ## Value is precise decimal. Ordinary math not allowed.
        self.updateText()
        self.skipValidate = False
        self.setCorrectionMode(self.CorrectToPreviousValue)
        self.setKeyboardTracking(False)
        self.proxy = SignalProxy(self.sigValueChanging, slot=self.delayedChange, delay=self.opts['delay'])
        self.setOpts(**kwargs)
        self._updateHeight()
        
        self.editingFinished.connect(self.editingFinishedEvent)

    def setOpts(self, **opts):
        """Set options affecting the behavior of the SpinBox.
        
        ============== ========================================================================
        **Arguments:**
        bounds         (min,max) Minimum and maximum values allowed in the SpinBox. 
                       Either may be None to leave the value unbounded. By default, values are
                       unbounded.
        suffix         (str) suffix (units) to display after the numerical value. By default,
                       suffix is an empty str.
        siPrefix       (bool) If True, then an SI prefix is automatically prepended
                       to the units and the value is scaled accordingly. For example,
                       if value=0.003 and suffix='V', then the SpinBox will display
                       "300 mV" (but a call to SpinBox.value will still return 0.003). In case
                       the value represents a dimensionless quantity that might span many
                       orders of magnitude, such as a Reynolds number, an SI
                       prefix is allowed with no suffix. Default is False.
        step           (float) The size of a single step. This is used when clicking the up/
                       down arrows, when rolling the mouse wheel, or when pressing 
                       keyboard arrows while the widget has keyboard focus. Note that
                       the interpretation of this value is different when specifying
                       the 'dec' argument. Default is 0.01.
        dec            (bool) If True, then the step value will be adjusted to match 
                       the current size of the variable (for example, a value of 15
                       might step in increments of 1 whereas a value of 1500 would
                       step in increments of 100). In this case, the 'step' argument
                       is interpreted *relative* to the current value. The most common
                       'step' values when dec=True are 0.1, 0.2, 0.5, and 1.0. Default is
                       False.
        minStep        (float) When dec=True, this specifies the minimum allowable step size.
        int            (bool) If True, the value is forced to integer type. Default is False
        finite         (bool) When False and int=False, infinite values (nan, inf, -inf) are
                       permitted. Default is True.
        wrapping       (bool) If True and both bounds are not None, spin box has circular behavior.
        decimals       (int) Number of decimal values to display. Default is 6. 
        format         (str) Formatting string used to generate the text shown. Formatting is
                       done with ``str.format()`` and makes use of several arguments:
                       
                       * *value* - the unscaled value of the spin box
                       * *suffix* - the suffix string
                       * *scaledValue* - the scaled value to use when an SI prefix is present
                       * *siPrefix* - the SI prefix string (if any), or an empty string if
                         this feature has been disabled
                       * *suffixGap* - a single space if a suffix is present, or an empty
                         string otherwise.
        regex          (str or RegexObject) Regular expression used to parse the spinbox text.
                       May contain the following group names:
                       
                       * *number* - matches the numerical portion of the string (mandatory)
                       * *siPrefix* - matches the SI prefix string
                       * *suffix* - matches the suffix string
                       
                       Default is defined in ``pyqtgraph.functions.FLOAT_REGEX``.
        evalFunc       (callable) Fucntion that converts a numerical string to a number,
                       preferrably a Decimal instance. This function handles only the numerical
                       of the text; it does not have access to the suffix or SI prefix.
        compactHeight  (bool) if True, then set the maximum height of the spinbox based on the
                       height of its font. This allows more compact packing on platforms with
                       excessive widget decoration. Default is True.
        ============== ========================================================================
        """
        #print opts
        for k,v in opts.items():
            if k == 'bounds':
                self.setMinimum(v[0], update=False)
                self.setMaximum(v[1], update=False)
            elif k == 'min':
                self.setMinimum(v, update=False)
            elif k == 'max':
                self.setMaximum(v, update=False)
            elif k in ['step', 'minStep']:
                self.opts[k] = D(asUnicode(v))
            elif k == 'value':
                pass   ## don't set value until bounds have been set
            elif k == 'format':
                self.opts[k] = asUnicode(v)
            elif k == 'regex' and isinstance(v, basestring):
                self.opts[k] = re.compile(v)
            elif k in self.opts:
                self.opts[k] = v
            else:
                raise TypeError("Invalid keyword argument '%s'." % k)
        if 'value' in opts:
            self.setValue(opts['value'])
            
        ## If bounds have changed, update value to match
        if 'bounds' in opts and 'value' not in opts:
            self.setValue()   
            
        ## sanity checks:
        if self.opts['int']:
            if 'step' in opts:
                step = opts['step']
                ## not necessary..
                #if int(step) != step:
                    #raise Exception('Integer SpinBox must have integer step size.')
            else:
                self.opts['step'] = int(self.opts['step'])
            
            if 'minStep' in opts:
                step = opts['minStep']
                if int(step) != step:
                    raise Exception('Integer SpinBox must have integer minStep size.')
            else:
                ms = int(self.opts.get('minStep', 1))
                if ms < 1:
                    ms = 1
                self.opts['minStep'] = ms

            if 'format' not in opts:
                self.opts['format'] = asUnicode("{value:d}{suffixGap}{suffix}")
        
        if 'delay' in opts:
            self.proxy.setDelay(opts['delay'])
        
        self.updateText()

    def setMaximum(self, m, update=True):
        """Set the maximum allowed value (or None for no limit)"""
        if m is not None:
            m = D(asUnicode(m))
        self.opts['bounds'][1] = m
        if update:
            self.setValue()
    
    def setMinimum(self, m, update=True):
        """Set the minimum allowed value (or None for no limit)"""
        if m is not None:
            m = D(asUnicode(m))
        self.opts['bounds'][0] = m
        if update:
            self.setValue()

    def wrapping(self):
        """Return whether or not the spin box is circular."""
        return self.opts['wrapping']

    def setWrapping(self, s):
        """Set whether spin box is circular.
        
        Both bounds must be set for this to have an effect."""
        self.opts['wrapping'] = s
        
    def setPrefix(self, p):
        """Set a string prefix.
        """
        self.setOpts(prefix=p)
    
    def setRange(self, r0, r1):
        """Set the upper and lower limits for values in the spinbox.
        """
        self.setOpts(bounds = [r0,r1])
        
    def setProperty(self, prop, val):
        ## for QSpinBox compatibility
        if prop == 'value':
            #if type(val) is QtCore.QVariant:
                #val = val.toDouble()[0]
            self.setValue(val)
        else:
            print("Warning: SpinBox.setProperty('%s', ..) not supported." % prop)

    def setSuffix(self, suf):
        """Set the string suffix appended to the spinbox text.
        """
        self.setOpts(suffix=suf)

    def setSingleStep(self, step):
        """Set the step size used when responding to the mouse wheel, arrow
        buttons, or arrow keys.
        """
        self.setOpts(step=step)
        
    def setDecimals(self, decimals):
        """Set the number of decimals to be displayed when formatting numeric
        values.
        """
        self.setOpts(decimals=decimals)
        
    def selectNumber(self):
        """
        Select the numerical portion of the text to allow quick editing by the user.
        """
        le = self.lineEdit()
        text = asUnicode(le.text())
        m = self.opts['regex'].match(text)
        if m is None:
            return
        s,e = m.start('number'), m.end('number')
        le.setSelection(s, e-s)

    def focusInEvent(self, ev):
        super(SpinBox, self).focusInEvent(ev)
        self.selectNumber()

    def value(self):
        """
        Return the value of this SpinBox.
        
        """
        if self.opts['int']:
            return int(self.val)
        else:
            return float(self.val)

    def setValue(self, value=None, update=True, delaySignal=False):
        """Set the value of this SpinBox.
        
        If the value is out of bounds, it will be clipped to the nearest boundary
        or wrapped if wrapping is enabled.
        
        If the spin is integer type, the value will be coerced to int.
        Returns the actual value set.
        
        If value is None, then the current value is used (this is for resetting
        the value after bounds, etc. have changed)
        """
        if value is None:
            value = self.value()

        bounded = True
        if not isnan(value):
            bounds = self.opts['bounds']
            if None not in bounds and self.opts['wrapping'] is True:
                bounded = False
                if isinf(value):
                    value = self.val
                else:
                    # Casting of Decimals to floats required to avoid unexpected behavior of remainder operator
                    value = float(value)
                    l, u = float(bounds[0]), float(bounds[1])
                    value = (value - l) % (u - l) + l
            else:
                if bounds[0] is not None and value < bounds[0]:
                    bounded = False
                    value = bounds[0]
                if bounds[1] is not None and value > bounds[1]:
                    bounded = False
                    value = bounds[1]

        if self.opts['int']:
            value = int(value)

        if not isinstance(value, D):
            value = D(asUnicode(value))

        prev, self.val = self.val, value
        changed = not fn.eq(value, prev)  # use fn.eq to handle nan

        if update and (changed or not bounded):
            self.updateText(prev=prev)

        if changed:
            self.sigValueChanging.emit(self, float(self.val))  ## change will be emitted in 300ms if there are no subsequent changes.
            if not delaySignal:
                self.emitChanged()

        return value
    
    def emitChanged(self):
        self.lastValEmitted = self.val
        self.valueChanged.emit(float(self.val))
        self.sigValueChanged.emit(self)
    
    def delayedChange(self):
        try:
            if not fn.eq(self.val, self.lastValEmitted):  # use fn.eq to handle nan
                self.emitChanged()
        except RuntimeError:
            pass  ## This can happen if we try to handle a delayed signal after someone else has already deleted the underlying C++ object.
    
    def widgetGroupInterface(self):
        return (self.valueChanged, SpinBox.value, SpinBox.setValue)
    
    def sizeHint(self):
        return QtCore.QSize(120, 0)
    
    def stepEnabled(self):
        return self.StepUpEnabled | self.StepDownEnabled        
    
    def stepBy(self, n):
        if isinf(self.val) or isnan(self.val):
            return

        n = D(int(n))   ## n must be integral number of steps.
        s = [D(-1), D(1)][n >= 0]  ## determine sign of step
        val = self.val
        
        for i in range(int(abs(n))):
            if self.opts['dec']:
                if val == 0:
                    step = self.opts['minStep']
                    exp = None
                else:
                    vs = [D(-1), D(1)][val >= 0]
                    #exp = D(int(abs(val*(D('1.01')**(s*vs))).log10()))
                    fudge = D('1.01')**(s*vs) ## fudge factor. at some places, the step size depends on the step sign.
                    exp = abs(val * fudge).log10().quantize(1, decimal.ROUND_FLOOR)
                    step = self.opts['step'] * D(10)**exp
                if 'minStep' in self.opts:
                    step = max(step, self.opts['minStep'])
                val += s * step
                #print "Exp:", exp, "step", step, "val", val
            else:
                val += s*self.opts['step']
                
            if 'minStep' in self.opts and abs(val) < self.opts['minStep']:
                val = D(0)
        self.setValue(val, delaySignal=True)  ## note all steps (arrow buttons, wheel, up/down keys..) emit delayed signals only.

    def valueInRange(self, value):
        if not isnan(value):
            bounds = self.opts['bounds']
            if bounds[0] is not None and value < bounds[0]:
                return False
            if bounds[1] is not None and value > bounds[1]:
                return False
            if self.opts.get('int', False):
                if int(value) != value:
                    return False
        return True

    def updateText(self, prev=None):
        # temporarily disable validation
        self.skipValidate = True
        
        txt = self.formatText(prev=prev)
        
        # actually set the text
        self.lineEdit().setText(txt)
        self.lastText = txt

        # re-enable the validation
        self.skipValidate = False
        
    def formatText(self, prev=None):
        # get the number of decimal places to print
        decimals = self.opts['decimals']
        suffix = self.opts['suffix']

        # format the string 
        val = self.value()
        if self.opts['siPrefix'] is True:
            # SI prefix was requested, so scale the value accordingly

            if self.val == 0 and prev is not None:
                # special case: if it's zero use the previous prefix
                (s, p) = fn.siScale(prev)
            else:
                (s, p) = fn.siScale(val)
            parts = {'value': val, 'suffix': suffix, 'decimals': decimals, 'siPrefix': p, 'scaledValue': s*val}

        else:
            # no SI prefix /suffix requested; scale is 1
            parts = {'value': val, 'suffix': suffix, 'decimals': decimals, 'siPrefix': '', 'scaledValue': val}

        parts['suffixGap'] = '' if (parts['suffix'] == '' and parts['siPrefix'] == '') else ' '
        
        return self.opts['format'].format(**parts)

    def validate(self, strn, pos):
        if self.skipValidate:
            ret = QtGui.QValidator.Acceptable
        else:
            try:
                val = self.interpret()
                if val is False:
                    ret = QtGui.QValidator.Intermediate
                else:
                    if self.valueInRange(val):
                        if not self.opts['delayUntilEditFinished']:
                            self.setValue(val, update=False)
                        ret = QtGui.QValidator.Acceptable
                    else:
                        ret = QtGui.QValidator.Intermediate
                        
            except:
                import sys
                sys.excepthook(*sys.exc_info())
                ret = QtGui.QValidator.Intermediate
            
        ## draw / clear border
        if ret == QtGui.QValidator.Intermediate:
            self.textValid = False
        elif ret == QtGui.QValidator.Acceptable:
            self.textValid = True
        ## note: if text is invalid, we don't change the textValid flag 
        ## since the text will be forced to its previous state anyway
        self.update()
        
        self.errorBox.setVisible(not self.textValid)
        
        ## support 2 different pyqt APIs. Bleh.
        if hasattr(QtCore, 'QString'):
            return (ret, pos)
        else:
            return (ret, strn, pos)
        
    def fixup(self, strn):
        # fixup is called when the spinbox loses focus with an invalid or intermediate string
        self.updateText()

        # support both PyQt APIs (for Python 2 and 3 respectively)
        # http://pyqt.sourceforge.net/Docs/PyQt4/python_v3.html#qvalidator
        try:
            strn.clear()
            strn.append(self.lineEdit().text())
        except AttributeError:
            return self.lineEdit().text()

    def interpret(self):
        """Return value of text or False if text is invalid."""
        strn = self.lineEdit().text()
        
        # tokenize into numerical value, si prefix, and suffix
        try:
            val, siprefix, suffix = fn.siParse(strn, self.opts['regex'], suffix=self.opts['suffix'])
        except Exception:
            return False
            
        # check suffix
        if suffix != self.opts['suffix']:
            return False
           
        # generate value
        val = self.opts['evalFunc'](val)

        if (self.opts['int'] or self.opts['finite']) and (isinf(val) or isnan(val)):
            return False

        if self.opts['int']:
            val = int(fn.siApply(val, siprefix))
        else:
            try:
                val = fn.siApply(val, siprefix)
            except Exception:
                import sys
                sys.excepthook(*sys.exc_info())
                return False

        return val

    def editingFinishedEvent(self):
        """Edit has finished; set value."""
        #print "Edit finished."
        if asUnicode(self.lineEdit().text()) == self.lastText:
            #print "no text change."
            return
        try:
            val = self.interpret()
        except Exception:
            return
        
        if val is False:
            #print "value invalid:", str(self.lineEdit().text())
            return
        if val == self.val:
            #print "no value change:", val, self.val
            return
        self.setValue(val, delaySignal=False)  ## allow text update so that values are reformatted pretty-like

    def _updateHeight(self):
        # SpinBox has very large margins on some platforms; this is a hack to remove those
        # margins and allow more compact packing of controls.
        if not self.opts['compactHeight']:
            self.setMaximumHeight(1e6)
            return
        h = QtGui.QFontMetrics(self.font()).height()
        if self._lastFontHeight != h:
            self._lastFontHeight = h
            self.setMaximumHeight(h)

    def paintEvent(self, ev):
        self._updateHeight()
        super().paintEvent(ev)


class ErrorBox(QtGui.QWidget):
    """Red outline to draw around lineedit when value is invalid.
    (for some reason, setting border from stylesheet does not work)
    """
    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        parent.installEventFilter(self)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._resize()
        self.setVisible(False)
        
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Resize:
            self._resize()
        return False

    def _resize(self):
        self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setPen(fn.mkPen(color='r', width=2))
        p.drawRect(self.rect())
        p.end()
