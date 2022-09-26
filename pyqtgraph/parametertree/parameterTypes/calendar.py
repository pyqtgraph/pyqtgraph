from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem


class CalendarParameterItem(WidgetParameterItem):
    def makeWidget(self):
        self.asSubItem = True
        w = QtWidgets.QCalendarWidget()
        w.setMaximumHeight(200)
        w.sigChanged = w.selectionChanged
        w.value = w.selectedDate
        w.setValue = w.setSelectedDate
        self.hideWidget = False
        self.param.opts.setdefault('default', QtCore.QDate.currentDate())
        return w


class CalendarParameter(Parameter):
    """
    Displays a Qt calendar whose date is specified by a 'format' option.

    ============== ========================================================
    **Options:**
    format         Format for displaying the date and converting from a string. Can be any value accepted by
                   `QDate.toString` and `fromString`, or a stringified version of a QDateFormat enum, i.e. 'ISODate',
                   'TextDate' (default), etc.
    ============== ========================================================
    """

    itemClass = CalendarParameterItem

    def __init__(self, **opts):
        opts.setdefault('format', 'TextDate')
        super().__init__(**opts)

    def _interpretFormat(self, fmt=None):
        fmt = fmt or self.opts.get('format')
        if hasattr(QtCore.Qt.DateFormat, fmt):
            fmt = getattr(QtCore.Qt.DateFormat, fmt)
        return fmt

    def _interpretValue(self, v):
        if isinstance(v, str):
            fmt = self._interpretFormat()
            if fmt is None:
                raise ValueError('Cannot parse date string without a set format')
            v = QtCore.QDate.fromString(v, fmt)
        return v

    def saveState(self, filter=None):
        state = super().saveState(filter)
        fmt = self._interpretFormat()
        if state['value'] is not None:
            state['value'] = state['value'].toString(fmt)
        return state
