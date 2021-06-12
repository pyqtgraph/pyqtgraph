# -*- coding: utf-8 -*-
import numpy as np
from ..Qt import QtGui, QtCore
from ..python2_3 import asUnicode, basestring
from .. import metaarray

translate = QtCore.QCoreApplication.translate

__all__ = ['TableWidget']


def _defersort(fn):
    def defersort(self, *args, **kwds):
        # may be called recursively; only the first call needs to block sorting
        setSorting = False
        if self._sorting is None:
            self._sorting = self.isSortingEnabled()
            setSorting = True
            self.setSortingEnabled(False)
        try:
            return fn(self, *args, **kwds)
        finally:
            if setSorting:
                self.setSortingEnabled(self._sorting)
                self._sorting = None
                
    return defersort


class TableWidget(QtGui.QTableWidget):
    """Extends QTableWidget with some useful functions for automatic data handling
    and copy / export context menu. Can automatically format and display a variety
    of data types (see :func:`setData() <pyqtgraph.TableWidget.setData>` for more
    information.
    """
    
    def __init__(self, *args, **kwds):
        """
        All positional arguments are passed to QTableWidget.__init__().
        
        ===================== =================================================
        **Keyword Arguments**
        editable              (bool) If True, cells in the table can be edited
                              by the user. Default is False.
        sortable              (bool) If True, the table may be soted by
                              clicking on column headers. Note that this also
                              causes rows to appear initially shuffled until
                              a sort column is selected. Default is True.
                              *(added in version 0.9.9)*
        ===================== =================================================
        """
        
        QtGui.QTableWidget.__init__(self, *args)
        
        self.itemClass = TableWidgetItem
        
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setSelectionMode(QtGui.QAbstractItemView.SelectionMode.ContiguousSelection)
        self.setSizePolicy(QtGui.QSizePolicy.Policy.Preferred, QtGui.QSizePolicy.Policy.Preferred)
        self.clear()
        
        kwds.setdefault('sortable', True)
        kwds.setdefault('editable', False)
        self.setEditable(kwds.pop('editable'))
        self.setSortingEnabled(kwds.pop('sortable'))
        
        if len(kwds) > 0:
            raise TypeError("Invalid keyword arguments '%s'" % list(kwds.keys()))
        
        self._sorting = None  # used when temporarily disabling sorting
        
        self._formats = {None: None} # stores per-column formats and entire table format
        self.sortModes = {} # stores per-column sort mode
        
        self.itemChanged.connect(self.handleItemChanged)
        
        self.contextMenu = QtGui.QMenu()
        self.contextMenu.addAction(translate("TableWidget", 'Copy Selection')).triggered.connect(self.copySel)
        self.contextMenu.addAction(translate("TableWidget", 'Copy All')).triggered.connect(self.copyAll)
        self.contextMenu.addAction(translate("TableWidget", 'Save Selection')).triggered.connect(self.saveSel)
        self.contextMenu.addAction(translate("TableWidget", 'Save All')).triggered.connect(self.saveAll)
        
    def clear(self):
        """Clear all contents from the table."""
        QtGui.QTableWidget.clear(self)
        self.verticalHeadersSet = False
        self.horizontalHeadersSet = False
        self.items = []
        self.setRowCount(0)
        self.setColumnCount(0)
        self.sortModes = {}
        
    def setData(self, data):
        """Set the data displayed in the table.
        Allowed formats are:
        
        * numpy arrays
        * numpy record arrays 
        * metaarrays
        * list-of-lists  [[1,2,3], [4,5,6]]
        * dict-of-lists  {'x': [1,2,3], 'y': [4,5,6]}
        * list-of-dicts  [{'x': 1, 'y': 4}, {'x': 2, 'y': 5}, ...]
        """
        self.clear()
        self.appendData(data)
        self.resizeColumnsToContents()
        
    @_defersort
    def appendData(self, data):
        """
        Add new rows to the table.
        
        See :func:`setData() <pyqtgraph.TableWidget.setData>` for accepted
        data types.
        """
        startRow = self.rowCount()
        
        fn0, header0 = self.iteratorFn(data)
        if fn0 is None:
            self.clear()
            return
        it0 = fn0(data)
        try:
            first = next(it0)
        except StopIteration:
            return
        fn1, header1 = self.iteratorFn(first)
        if fn1 is None:
            self.clear()
            return
        
        firstVals = [x for x in fn1(first)]
        self.setColumnCount(len(firstVals))
        
        if not self.verticalHeadersSet and header0 is not None:
            labels = [self.verticalHeaderItem(i).text() for i in range(self.rowCount())]
            self.setRowCount(startRow + len(header0))
            self.setVerticalHeaderLabels(labels + header0)
            self.verticalHeadersSet = True
        if not self.horizontalHeadersSet and header1 is not None:
            self.setHorizontalHeaderLabels(header1)
            self.horizontalHeadersSet = True
        
        i = startRow
        self.setRow(i, firstVals)
        for row in it0:
            i += 1
            self.setRow(i, [x for x in fn1(row)])
            
        if (self._sorting and self.horizontalHeadersSet and 
            self.horizontalHeader().sortIndicatorSection() >= self.columnCount()):
            self.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
    
    def setEditable(self, editable=True):
        self.editable = editable
        for item in self.items:
            item.setEditable(editable)
    
    def setFormat(self, format, column=None):
        """
        Specify the default text formatting for the entire table, or for a
        single column if *column* is specified.
        
        If a string is specified, it is used as a format string for converting
        float values (and all other types are converted using str). If a 
        function is specified, it will be called with the item as its only
        argument and must return a string. Setting format = None causes the 
        default formatter to be used instead.
        
        Added in version 0.9.9.
        
        """
        if format is not None and not isinstance(format, basestring) and not callable(format):
            raise ValueError("Format argument must string, callable, or None. (got %s)" % format)
        
        self._formats[column] = format
        
        
        if column is None:
            # update format of all items that do not have a column format 
            # specified
            for c in range(self.columnCount()):
                if self._formats.get(c, None) is None:
                    for r in range(self.rowCount()):
                        item = self.item(r, c)
                        if item is None:
                            continue
                        item.setFormat(format)
        else:
            # set all items in the column to use this format, or the default 
            # table format if None was specified.
            if format is None:
                format = self._formats[None]
            for r in range(self.rowCount()):
                item = self.item(r, column)
                if item is None:
                    continue
                item.setFormat(format)
        
    
    def iteratorFn(self, data):
        ## Return 1) a function that will provide an iterator for data and 2) a list of header strings
        if isinstance(data, list) or isinstance(data, tuple):
            return lambda d: d.__iter__(), None
        elif isinstance(data, dict):
            return lambda d: iter(d.values()), list(map(asUnicode, data.keys()))
        elif (hasattr(data, 'implements') and data.implements('MetaArray')):
            if data.axisHasColumns(0):
                header = [asUnicode(data.columnName(0, i)) for i in range(data.shape[0])]
            elif data.axisHasValues(0):
                header = list(map(asUnicode, data.xvals(0)))
            else:
                header = None
            return self.iterFirstAxis, header
        elif isinstance(data, np.ndarray):
            return self.iterFirstAxis, None
        elif isinstance(data, np.void):
            return self.iterate, list(map(asUnicode, data.dtype.names))
        elif data is None:
            return (None,None)
        elif np.isscalar(data):
            return self.iterateScalar, None
        else:
            msg = "Don't know how to iterate over data type: {!s}".format(type(data))
            raise TypeError(msg)
        
    def iterFirstAxis(self, data):
        for i in range(data.shape[0]):
            yield data[i]
            
    def iterate(self, data):
        # for numpy.void, which can be iterated but mysteriously 
        # has no __iter__ (??)
        for x in data:
            yield x
        
    def iterateScalar(self, data):
        yield data
        
    def appendRow(self, data):
        self.appendData([data])
        
    @_defersort
    def addRow(self, vals):
        row = self.rowCount()
        self.setRowCount(row + 1)
        self.setRow(row, vals)
        
    @_defersort
    def setRow(self, row, vals):
        if row > self.rowCount() - 1:
            self.setRowCount(row + 1)
        for col in range(len(vals)):
            val = vals[col]
            item = self.itemClass(val, row)
            item.setEditable(self.editable)
            sortMode = self.sortModes.get(col, None)
            if sortMode is not None:
                item.setSortMode(sortMode)
            format = self._formats.get(col, self._formats[None])
            item.setFormat(format)
            self.items.append(item)
            self.setItem(row, col, item)
            item.setValue(val)  # Required--the text-change callback is invoked
                                # when we call setItem.

    def setSortMode(self, column, mode):
        """
        Set the mode used to sort *column*.
        
        ============== ========================================================
        **Sort Modes**
        value          Compares item.value if available; falls back to text
                       comparison.
        text           Compares item.text()
        index          Compares by the order in which items were inserted.
        ============== ========================================================
        
        Added in version 0.9.9
        """
        for r in range(self.rowCount()):
            item = self.item(r, column)
            if hasattr(item, 'setSortMode'):
                item.setSortMode(mode)
        self.sortModes[column] = mode
        
    def sizeHint(self):
        # based on http://stackoverflow.com/a/7195443/54056
        width = sum(self.columnWidth(i) for i in range(self.columnCount()))
        width += self.verticalHeader().sizeHint().width()
        width += self.verticalScrollBar().sizeHint().width()
        width += self.frameWidth() * 2
        height = sum(self.rowHeight(i) for i in range(self.rowCount()))
        height += self.verticalHeader().sizeHint().height()
        height += self.horizontalScrollBar().sizeHint().height()
        return QtCore.QSize(width, height)
         
    def serialize(self, useSelection=False):
        """Convert entire table (or just selected area) into tab-separated text values"""
        if useSelection:
            selection = self.selectedRanges()[0]
            rows = list(range(selection.topRow(),
                              selection.bottomRow() + 1))
            columns = list(range(selection.leftColumn(),
                                 selection.rightColumn() + 1))        
        else:
            rows = list(range(self.rowCount()))
            columns = list(range(self.columnCount()))

        data = []
        if self.horizontalHeadersSet:
            row = []
            if self.verticalHeadersSet:
                row.append(asUnicode(''))
            
            for c in columns:
                row.append(asUnicode(self.horizontalHeaderItem(c).text()))
            data.append(row)
        
        for r in rows:
            row = []
            if self.verticalHeadersSet:
                row.append(asUnicode(self.verticalHeaderItem(r).text()))
            for c in columns:
                item = self.item(r, c)
                if item is not None:
                    row.append(asUnicode(item.value))
                else:
                    row.append(asUnicode(''))
            data.append(row)
            
        s = ''
        for row in data:
            s += ('\t'.join(row) + '\n')
        return s

    def copySel(self):
        """Copy selected data to clipboard."""
        QtGui.QApplication.clipboard().setText(self.serialize(useSelection=True))

    def copyAll(self):
        """Copy all data to clipboard."""
        QtGui.QApplication.clipboard().setText(self.serialize(useSelection=False))

    def saveSel(self):
        """Save selected data to file."""
        self.save(self.serialize(useSelection=True))

    def saveAll(self):
        """Save all data to file."""
        self.save(self.serialize(useSelection=False))

    def save(self, data):
        fileName = QtGui.QFileDialog.getSaveFileName(
            self,
            f"{translate('TableWidget', 'Save As')}...",
            "",
            f"{translate('TableWidget', 'Tab-separated values')} (*.tsv)"
        )
        if isinstance(fileName, tuple):
            fileName = fileName[0]  # Qt4/5 API difference
        if fileName == '':
            return
        with open(fileName, 'w') as fd:
            fd.write(data)

    def contextMenuEvent(self, ev):
        self.contextMenu.popup(ev.globalPos())
        
    def keyPressEvent(self, ev):
        if ev.matches(QtGui.QKeySequence.StandardKey.Copy):
            ev.accept()
            self.copySel()
        else:
            super().keyPressEvent(ev)

    def handleItemChanged(self, item):
        item.itemChanged()


class TableWidgetItem(QtGui.QTableWidgetItem):
    def __init__(self, val, index, format=None):
        QtGui.QTableWidgetItem.__init__(self, '')
        self._blockValueChange = False
        self._format = None
        self._defaultFormat = '%0.3g'
        self.sortMode = 'value'
        self.index = index
        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        self.setFlags(flags)
        self.setValue(val)
        self.setFormat(format)
        
    def setEditable(self, editable):
        """
        Set whether this item is user-editable.
        """
        if editable:
            self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        else:
            self.setFlags(self.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            
    def setSortMode(self, mode):
        """
        Set the mode used to sort this item against others in its column.
        
        ============== ========================================================
        **Sort Modes**
        value          Compares item.value if available; falls back to text
                       comparison.
        text           Compares item.text()
        index          Compares by the order in which items were inserted.
        ============== ========================================================
        """
        modes = ('value', 'text', 'index', None)
        if mode not in modes:
            raise ValueError('Sort mode must be one of %s' % str(modes))
        self.sortMode = mode
        
    def setFormat(self, fmt):
        """Define the conversion from item value to displayed text. 
        
        If a string is specified, it is used as a format string for converting
        float values (and all other types are converted using str). If a 
        function is specified, it will be called with the item as its only
        argument and must return a string.
        
        Added in version 0.9.9.
        """
        if fmt is not None and not isinstance(fmt, basestring) and not callable(fmt):
            raise ValueError("Format argument must string, callable, or None. (got %s)" % fmt)
        self._format = fmt
        self._updateText()
        
    def _updateText(self):
        self._blockValueChange = True
        try:
            self._text = self.format()
            self.setText(self._text)
        finally:
            self._blockValueChange = False

    def setValue(self, value):
        self.value = value
        self._updateText()

    def itemChanged(self):
        """Called when the data of this item has changed."""
        if self.text() != self._text:
            self.textChanged()

    def textChanged(self):
        """Called when this item's text has changed for any reason."""
        self._text = self.text()

        if self._blockValueChange:
            # text change was result of value or format change; do not
            # propagate.
            return
        
        try:

            self.value = type(self.value)(self.text())
        except ValueError:
            self.value = str(self.text())

    def format(self):
        if callable(self._format):
            return self._format(self)
        if isinstance(self.value, (float, np.floating)):
            if self._format is None:
                return self._defaultFormat % self.value
            else:
                return self._format % self.value
        else:
            return asUnicode(self.value)

    def __lt__(self, other):
        if self.sortMode == 'index' and hasattr(other, 'index'):
            return self.index < other.index
        if self.sortMode == 'value' and hasattr(other, 'value'):
            return self.value < other.value
        else:
            return self.text() < other.text()


if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    t = TableWidget()
    win.setCentralWidget(t)
    win.resize(800,600)
    win.show()
    
    ll = [[1,2,3,4,5]] * 20
    ld = [{'x': 1, 'y': 2, 'z': 3}] * 20
    dl = {'x': list(range(20)), 'y': list(range(20)), 'z': list(range(20))}
    
    a = np.ones((20, 5))
    ra = np.ones((20,), dtype=[('x', int), ('y', int), ('z', int)])
    
    t.setData(ll)
    
    ma = metaarray.MetaArray(np.ones((20, 3)), info=[
        {'values': np.linspace(1, 5, 20)}, 
        {'cols': [
            {'name': 'x'},
            {'name': 'y'},
            {'name': 'z'},
        ]}
    ])
    t.setData(ma)
    
