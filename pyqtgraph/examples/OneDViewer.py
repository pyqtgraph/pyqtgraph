"""
Prototype 1D signal viewer with MATLAB-like curve selection and data editing.

- Click a curve (or its entry in the trace list on the left) to select it.
- Shift-click / Ctrl-click a curve, or Ctrl/Shift-click in the list, to extend
  or toggle a multi-selection.
- Ctrl+C / Ctrl+X / Ctrl+V copy / cut / paste the selected curve(s). Copy /
  cut write the data to the *system* clipboard as tab-separated text (so it
  can be pasted into Excel/MATLAB/a text editor), plus a private metadata
  format carrying each curve's pen (color/width/style). Paste reads both back
  — so copying in one viewer window and pasting into another (via "New
  Viewer" / Ctrl+N) works, including across separate instances of this
  script, and keeps the original curve style.
- The console at the bottom exposes `selections` (the list of selected curve
  names, in selection order), `selection` (just the first one, or None if
  nothing's selected — a shortcut for the common single-curve case), and
  `data` (a dict of the selected curves' data keyed by name, e.g.
  `data[selection].y`). Edit the arrays, then call apply() to push the
  changes back onto the plot.
- The "Curve style" panel (bottom-left) edits the color/width/line-style of
  the selected curve(s) via a ParameterTree.
- Ctrl+O (or right-click the plot -> Import..., right next to the plot's
  own Export... action — every pyqtgraph GraphicsScene has this built in)
  opens pyqtgraph's own ImportDialog: pick a file, preview the curves found
  in it, uncheck any you don't want, and import. Its CSVImporter
  understands Export... > "CSV of original plot data" (comma or tab
  delimited, one x column per curve or one shared x column, named or
  unnamed columns), and falls back to a plain whitespace/comma two-column
  x,y file with no header. For a full round-trip (data + curve name/pen +
  axis labels/title), use Export... > "pyqtgraph Plot (JSON, data +
  style)" and its matching PlotProjectImporter instead — it reconstructs
  the whole figure, not just the numbers. Curves added this way (or by any
  other code that pokes at the plot directly) are picked up automatically
  so they stay
  selectable/copyable like any other curve.
"""

import json

import numpy as np

import pyqtgraph as pg
import pyqtgraph.console
import pyqtgraph.importers
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

HIGHLIGHT_WIDTH_BOOST = 2

# Private MIME type carrying per-curve style, alongside the plain-text CSV
# that external apps (Excel, MATLAB, a text editor) can read.
PEN_META_MIME_TYPE = "application/x-pyqtgraph-onedviewer-pens+json"


def _enum_int(e):
    """Cross-binding enum -> int (PyQt6/PySide6 need .value; others don't)."""
    return int(getattr(e, 'value', e))


def _pen_to_dict(pen):
    pen = pg.mkPen(pen)
    return {
        'rgba': pen.color().getRgb(),  # (r, g, b, a), each 0-255
        'width': pen.widthF(),
        'style': _enum_int(pen.style()),
    }


def _pen_from_dict(d):
    pen = pg.mkPen(color=tuple(d['rgba']), width=d.get('width', 1))
    if d.get('style') is not None:
        pen.setStyle(QtCore.Qt.PenStyle(d['style']))
    return pen


class CurveData:
    """Console-facing view of one curve's data, keyed by curve name in the
    `data` dict exposed to the console (e.g. `data['sine'].y`)."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"CurveData(x=array[{len(self.x)}], y=array[{len(self.y)}])"


class Clipboard:
    """Converts curve data (and, via a private MIME type, curve style) to/from
    the system clipboard, so copy/paste works across viewer windows and
    processes. The plain-text CSV part stays readable by external apps.
    """

    @staticmethod
    def to_csv_text(entries):
        if not entries:
            return ""
        header = []
        columns = []
        for entry in entries:
            header += [f"{entry['name']}_x", f"{entry['name']}_y"]
            columns.append(entry['x'])
            columns.append(entry['y'])
        lines = ["\t".join(header)]
        n = max(len(c) for c in columns)
        for i in range(n):
            row = [str(c[i]) if i < len(c) else "" for c in columns]
            lines.append("\t".join(row))
        return "\n".join(lines)

    @staticmethod
    def from_csv_text(text):
        """Parse data written either by `to_csv_text` or by pyqtgraph's own
        CSVExporter/CSVImporter multi-curve format (see
        `pyqtgraph.importers.CSVImporter.parseCurvesFromText`)."""
        return pyqtgraph.importers.parseCurvesFromText(text)

    @staticmethod
    def to_mime_data(entries):
        mime = QtCore.QMimeData()
        mime.setText(Clipboard.to_csv_text(entries))
        pens_meta = [{'name': e['name'], 'pen': _pen_to_dict(e['pen'])} for e in entries]
        mime.setData(PEN_META_MIME_TYPE, QtCore.QByteArray(json.dumps(pens_meta).encode('utf-8')))
        return mime

    @staticmethod
    def from_mime_data(mime):
        entries = Clipboard.from_csv_text(mime.text())
        if not entries:
            return []
        pens_by_name = {}
        if mime.hasFormat(PEN_META_MIME_TYPE):
            try:
                pens_meta = json.loads(bytes(mime.data(PEN_META_MIME_TYPE)).decode('utf-8'))
                pens_by_name = {m['name']: _pen_from_dict(m['pen']) for m in pens_meta}
            except (ValueError, KeyError):
                pens_by_name = {}
        for entry in entries:
            entry['pen'] = pens_by_name.get(entry['name'], pg.mkPen(width=1))
        return entries


class OneDViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyqtgraph example: OneDViewer")

        self.selected = []  # list of PlotDataItem, in selection order
        self._orig_pens = {}  # PlotDataItem -> original pen
        self._items_by_name = {}
        self._paste_count = 0
        self._load_count = 0
        self._updating_pen_param = False
        self._sync_scheduled = False

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.setLabel('bottom', 'x')
        self.plot.setLabel('left', 'y')

        # Every GraphicsScene already offers "Import..." right next to
        # "Export..." in its right-click menu (see pyqtgraph.GraphicsScene /
        # pyqtgraph.importers). Curves it adds land directly on this
        # PlotItem, so pick them up automatically to keep them
        # selectable/copyable like any curve added via addCurve().
        self.plot.scene().changed.connect(self._onSceneChanged)

        self.traceList = QtWidgets.QListWidget()
        self.traceList.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.traceList.itemSelectionChanged.connect(self._onListSelectionChanged)

        newViewerButton = QtWidgets.QPushButton("New Viewer")
        newViewerButton.setToolTip("Ctrl+N")
        newViewerButton.clicked.connect(lambda: openNewViewer())

        buttonRow = QtWidgets.QHBoxLayout()
        buttonRow.addWidget(newViewerButton)

        self.styleParams = ptree.Parameter.create(
            name='params', type='group',
            children=[dict(name='pen', title='Curve style', type='pen', value=pg.mkPen(width=1))],
        )
        self.styleParams.child('pen').sigValueChanged.connect(self._onPenParamChanged)
        self.styleParams.child('pen').setOpts(enabled=False)
        self.penTree = ptree.ParameterTree(showHeader=False)
        self.penTree.setParameters(self.styleParams, showTop=False)
        self.penTree.setMaximumHeight(120)

        leftPanel = QtWidgets.QWidget()
        leftLayout = QtWidgets.QVBoxLayout(leftPanel)
        leftLayout.setContentsMargins(0, 0, 0, 0)
        leftLayout.addLayout(buttonRow)
        leftLayout.addWidget(self.traceList)
        leftLayout.addWidget(self.penTree)

        topSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        topSplitter.addWidget(leftPanel)
        topSplitter.addWidget(self.plot)
        topSplitter.setStretchFactor(0, 1)
        topSplitter.setStretchFactor(1, 4)

        self.console = pyqtgraph.console.ConsoleWidget(
            namespace={'np': np, 'apply': self.applyFromConsole},
            text=(
                "Select one or more curves (click a line, or click in the list "
                "on the left; Shift/Ctrl extend the selection).\n"
                "'selections' holds all their names, 'selection' just the first one "
                "(or None). 'data' holds their arrays, e.g. data[selection].y\n"
                "Edit the arrays, then call apply() to push the changes back to the plot."
            ),
        )

        mainSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        mainSplitter.addWidget(topSplitter)
        mainSplitter.addWidget(self.console)
        mainSplitter.setStretchFactor(0, 3)
        mainSplitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(mainSplitter)

        self._makeShortcuts()

    # ---- adding data ----------------------------------------------------
    def addCurve(self, x, y, name):
        item = self.plot.plot(x, y, name=name, pen=pg.mkPen(width=1))
        self._registerCurve(item, name)
        return item

    def _registerCurve(self, item, name):
        item.setCurveClickable(True, width=8)
        item.sigClicked.connect(lambda _clicked, ev, item=item: self._onCurveClicked(item, ev))
        item.curveName = name
        self._items_by_name[name] = item
        self.traceList.addItem(name)

    def _onSceneChanged(self, *args):
        # Coalesce bursts (e.g. during pan/zoom) into one deferred sync.
        if self._sync_scheduled:
            return
        self._sync_scheduled = True
        QtCore.QTimer.singleShot(0, self._syncTrackedCurves)

    def _syncTrackedCurves(self):
        """Adopt curves added directly to the plot by something other than
        addCurve() — e.g. the built-in right-click Import... action — so
        they're selectable/copyable/stylable like any other curve here."""
        self._sync_scheduled = False
        tracked = set(self._items_by_name.values())
        for curve in self.plot.getPlotItem().curves:
            if curve in tracked:
                continue
            name = curve.name() or f"curve{len(self._items_by_name)}"
            while name in self._items_by_name:
                name = f"{name}_2"
            self._registerCurve(curve, name)

    # ---- selection --------------------------------------------------------
    def _onCurveClicked(self, item, ev):
        modifiers = ev.modifiers()
        extend = bool(
            modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
            or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        )
        if extend:
            self._toggleSelect(item)
        else:
            self._setSelection([item])
        self._syncListFromSelection()

    def _onListSelectionChanged(self):
        names = [i.text() for i in self.traceList.selectedItems()]
        items = [self._items_by_name[n] for n in names]
        self._setSelectionQuiet(items)

    def _toggleSelect(self, item):
        if item in self.selected:
            self._setSelection([i for i in self.selected if i is not item])
        else:
            self._setSelection(self.selected + [item])

    def _setSelection(self, items):
        self._setSelectionQuiet(items)
        self._syncListFromSelection()

    def _setSelectionQuiet(self, items):
        for item in self.selected:
            if item not in items and item in self._orig_pens:
                item.setPen(self._orig_pens.pop(item))
        for item in items:
            if item not in self._orig_pens:
                self._orig_pens[item] = item.opts['pen']
                pen = pg.mkPen(item.opts['pen'])
                pen.setWidth(pen.width() + HIGHLIGHT_WIDTH_BOOST)
                item.setPen(pen)
        self.selected = list(items)
        self._updateConsoleNamespace()
        self._populatePenParams()

    def _syncListFromSelection(self):
        self.traceList.blockSignals(True)
        self.traceList.clearSelection()
        for item in self.selected:
            for m in self.traceList.findItems(item.curveName, QtCore.Qt.MatchFlag.MatchExactly):
                m.setSelected(True)
        self.traceList.blockSignals(False)

    # ---- console ------------------------------------------------------
    def _updateConsoleNamespace(self):
        ns = self.console.localNamespace
        ns['selections'] = [item.curveName for item in self.selected]
        ns['selection'] = ns['selections'][0] if ns['selections'] else None
        ns['data'] = {item.curveName: CurveData(item.xData, item.yData) for item in self.selected}

    def applyFromConsole(self):
        data = self.console.localNamespace.get('data', {})
        applied = 0
        for item in self.selected:
            curve = data.get(item.curveName)
            if curve is None:
                continue
            item.setData(curve.x, curve.y)
            applied += 1
        return f"applied to {applied} curve(s)"

    # ---- curve style (ParameterTree) ---------------------------------
    def _populatePenParams(self):
        penParam = self.styleParams.child('pen')
        if not self.selected:
            penParam.setOpts(enabled=False)
            return
        penParam.setOpts(enabled=True)
        self._updating_pen_param = True
        try:
            penParam.setValue(self._basePen(self.selected[0]))
        finally:
            self._updating_pen_param = False

    def _onPenParamChanged(self, param, pen):
        if self._updating_pen_param or not self.selected:
            return
        for item in self.selected:
            self._setCurvePen(item, pg.mkPen(pen))

    def _setCurvePen(self, item, pen):
        if item in self._orig_pens:
            self._orig_pens[item] = pen
            highlighted = pg.mkPen(pen)
            highlighted.setWidth(highlighted.width() + HIGHLIGHT_WIDTH_BOOST)
            item.setPen(highlighted)
        else:
            item.setPen(pen)

    # ---- loading data from a file --------------------------------------
    def loadDataDialog(self):
        """Open pyqtgraph's own (core) ImportDialog, targeting this plot."""
        self.plot.scene().showImportDialog(item=self.plot.getPlotItem())

    def loadDataFile(self, path):
        """Load a file directly, skipping the picker dialog (for scripting)."""
        try:
            entries = pyqtgraph.importers.parseCurvesFromFile(path)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", f"Could not load {path}:\n{e}")
            return
        self._addLoadedEntries(entries)

    def _addLoadedEntries(self, entries):
        new_items = []
        for entry in entries:
            name = entry['name']
            while name in self._items_by_name:
                self._load_count += 1
                name = f"{entry['name']}_{self._load_count}"
            item = self.addCurve(entry['x'], entry['y'], name)
            new_items.append(item)
        self._setSelection(new_items)

    # ---- clipboard ------------------------------------------------------
    def _makeShortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, self, activated=self.copySelection)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Cut, self, activated=self.cutSelection)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Paste, self, activated=self.pasteClipboard)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.New, self, activated=lambda: openNewViewer())
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self, activated=self.loadDataDialog)

    def _basePen(self, item):
        """The curve's real pen, ignoring the temporary selection highlight."""
        return self._orig_pens.get(item, item.opts['pen'])

    def copySelection(self):
        if not self.selected:
            return
        entries = [
            dict(
                name=item.curveName,
                x=np.array(item.xData),
                y=np.array(item.yData),
                pen=self._basePen(item),
            )
            for item in self.selected
        ]
        QtWidgets.QApplication.clipboard().setMimeData(Clipboard.to_mime_data(entries))

    def cutSelection(self):
        if not self.selected:
            return
        self.copySelection()
        for item in list(self.selected):
            self.plot.removeItem(item)
            del self._items_by_name[item.curveName]
            for m in self.traceList.findItems(item.curveName, QtCore.Qt.MatchFlag.MatchExactly):
                self.traceList.takeItem(self.traceList.row(m))
        self._setSelection([])

    def pasteClipboard(self):
        mime = QtWidgets.QApplication.clipboard().mimeData()
        entries = Clipboard.from_mime_data(mime)
        if not entries:
            return
        new_items = []
        for entry in entries:
            name = entry['name']
            while name in self._items_by_name:
                self._paste_count += 1
                name = f"{entry['name']}_copy{self._paste_count}"
            item = self.addCurve(entry['x'], entry['y'], name)
            item.setPen(entry['pen'])
            new_items.append(item)
        self._setSelection(new_items)


def _demoData():
    x = np.linspace(0, 10, 500)
    return [
        (x, np.sin(x), 'sine'),
        (x, np.cos(x), 'cosine'),
        (x, 0.3 * np.sin(3 * x) + 0.5, 'harmonic'),
    ]


_open_windows = []  # keep references so windows aren't garbage-collected


def openNewViewer():
    """Open a new, initially empty OneDViewer window (paste into it with Ctrl+V)."""
    win = OneDViewer()
    win.resize(1000, 700)
    win.show()
    _open_windows.append(win)
    return win


if __name__ == '__main__':
    app = pg.mkQApp("OneDViewer example")
    win = openNewViewer()
    for x, y, name in _demoData():
        win.addCurve(x, y, name)
    pg.exec()
