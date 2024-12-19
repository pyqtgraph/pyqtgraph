from qtpy import QtWidgets, QtCore
from qtpy.QtCore import QDateTime, QTime
from pyqtgraph import functions as fn
from .basetypes import WidgetParameterItem
from .basetypes import SimpleParameter
from pyqtgraph.parametertree import ParameterItem
from ..xml_parameter_factory import XMLParameter

import numpy as np


class WidgetParameterItem(WidgetParameterItem):

    def updateDisplayLabel(self, value=None):
        """Update the display label to reflect the value of the parameter."""
        if value is None:
            value = self.param.value()
        self.displayLabel.setText(value.toString(self.param.opts['format']))


class DateParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def makeWidget(self):
        opts = self.param.opts
        w = QtWidgets.QDateEdit(QtCore.QDate(QtCore.QDate.currentDate()))
        w.setCalendarPopup(True)
        if 'format' not in opts:
            opts['format'] = 'dd/MM/yyyy'
        w.setDisplayFormat(opts['format'])

        w.sigChanged = w.dateChanged
        w.value = w.date
        w.setValue = w.setDate
        return w


class DateTimeParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def makeWidget(self):
        opts = self.param.opts
        w = QtWidgets.QDateTimeEdit(QtCore.QDateTime(QtCore.QDate.currentDate(),
                                                     QtCore.QTime.currentTime()))
        w.setCalendarPopup(True)
        if 'format' not in opts:
            opts['format'] = 'dd/MM/yyyy hh:mm'
        w.setDisplayFormat(opts['format'])
        w.sigChanged = w.dateTimeChanged
        w.value = w.dateTime
        w.setValue = w.setDateTime
        return w


class QTimeCustom(QtWidgets.QTimeEdit):
    def __init__(self, *args, **kwargs):
        super(QTimeCustom, self).__init__(*args, **kwargs)
        self.minutes_increment = 1
        self.timeChanged.connect(self.updateTime)

    def setTime(self, time):
        hours = time.hour()
        minutes = time.minute()

        minutes = int(np.round(minutes / self.minutes_increment) * self.minutes_increment)
        if minutes == 60:
            minutes = 0
            hours += 1

        time.setHMS(hours, minutes, 0)

        return super(QTimeCustom, self).setTime(time)

    def setMinuteIncrement(self, minutes_increment):
        self.minutes_increment = minutes_increment
        self.updateTime(self.time())

    @QtCore.Slot(QtCore.QTime)
    def updateTime(self, time):
        self.setTime(time)


class TimeParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def makeWidget(self):
        opts = self.param.opts
        w = QTimeCustom(QtCore.QTime(QtCore.QTime.currentTime()))
        if 'minutes_increment' in opts:
            w.setMinuteIncrement(opts['minutes_increment'])
        if 'format' not in opts:
            opts['format'] = 'hh:mm'
        w.setDisplayFormat(opts['format'])

        w.sigChanged = w.timeChanged
        w.value = w.time
        w.setValue = w.setTime
        return w

    def optsChanged(self, param, opts):
        """

        """
        super().optsChanged(param, opts)
        if 'minutes_increment' in opts:
            self.widget.setMinuteIncrement(opts['minutes_increment'])


class DateParameter(SimpleParameter, XMLParameter):
    itemClass = DateParameterItem

    def _interpretValue(self, v):
        return QtCore.QDate(v)
    
    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','0')
        param_dict['value'] = QDateTime.fromMSecsSinceEpoch(int(value)).date()

        return param_dict
        
    @staticmethod
    def get_specific_options(param):
        param_value = param.opts.get('value', None)
        opts = {
            "value": str(QDateTime(param_value, QTime()).toMSecsSinceEpoch())
        }

        return opts


class DateTimeParameter(SimpleParameter, XMLParameter):
    itemClass = DateTimeParameterItem

    def _interpretValue(self, v):
        return QtCore.QDateTime(v)

    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','0')
        param_dict['value'] = QDateTime.fromMSecsSinceEpoch(int(value))

        return param_dict

    @staticmethod   
    def get_specific_options(param):
        param_value = param.opts.get('value', None)
        opts = {
            "value": str(param_value.toMSecsSinceEpoch()),
        }
        
        return opts
    

class TimeParameter(SimpleParameter):
    itemClass = TimeParameterItem

    def _interpretValue(self, v):
        if isinstance(v, QtCore.QTime):
            return v
        elif isinstance(v, str):
            return QtCore.QTime(*eval(v.split('QTime')[1]))
        