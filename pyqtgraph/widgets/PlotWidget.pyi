"""
Type stub for pyqtgraph.widgets.PlotWidget
Contributes to: https://github.com/pyqtgraph/pyqtgraph/issues/3481

Based on PEP 484 and PEP 561.
"""

from typing import Any, Optional, Sequence

from ..Qt import QtCore, QtWidgets
from ..graphicsItems.PlotItem import PlotItem
from .GraphicsView import GraphicsView

__all__: list[str]

class PlotWidget(GraphicsView):
    """
    GraphicsView widget with a single PlotItem inside.

    The following methods are wrapped directly from PlotItem / ViewBox:
    addItem, removeItem, clear, setAxisItems,
    setXRange, setYRange, setRange, autoRange,
    setXLink, setYLink, viewRect,
    setMouseEnabled, enableAutoRange, disableAutoRange,
    setAspectLocked, setLimits, register, unregister.

    For all other methods, use getPlotItem().
    """

    # Signals
    sigRangeChanged: QtCore.SignalInstance   # emits (PlotWidget, range)
    sigTransformChanged: QtCore.SignalInstance

    # The contained PlotItem (may be None briefly during __init__)
    plotItem: PlotItem

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        background: Any = "default",
        plotItem: Optional[PlotItem] = None,
        **kargs: Any,
    ) -> None: ...

    def close(self) -> None: ...

    def __getattr__(self, attr: str) -> Any: ...

    @QtCore.Slot(object, object)
    def viewRangeChanged(self, view: Any, range: Any) -> None: ...

    def widgetGroupInterface(self) -> tuple[None, Any, Any]: ...

    def saveState(self) -> dict[str, Any]: ...

    def restoreState(self, state: dict[str, Any]) -> None: ...

    def getPlotItem(self) -> PlotItem:
        """Return the PlotItem contained within."""
        ...

    # ------------------------------------------------------------------ #
    # Methods wrapped from PlotItem / ViewBox at runtime via setattr.
    # Static type checkers cannot see dynamic wrapping, so we declare
    # them explicitly here.
    # ------------------------------------------------------------------ #

    def addItem(self, item: Any, *args: Any, **kargs: Any) -> None: ...

    def removeItem(self, item: Any) -> None: ...

    def clear(self) -> None: ...

    def setAxisItems(self, axisItems: Optional[dict[str, Any]] = None) -> None: ...

    def setXRange(
        self,
        min: float,
        max: float,
        padding: Optional[float] = None,
        update: bool = True,
    ) -> None: ...

    def setYRange(
        self,
        min: float,
        max: float,
        padding: Optional[float] = None,
        update: bool = True,
    ) -> None: ...

    def setRange(
        self,
        rect: Any = None,
        xRange: Optional[Sequence[float]] = None,
        yRange: Optional[Sequence[float]] = None,
        padding: Optional[float] = None,
        update: bool = True,
        disableAutoRange: bool = True,
    ) -> None: ...

    def autoRange(
        self,
        padding: Optional[float] = None,
        items: Optional[list[Any]] = None,
        item: Any = None,
    ) -> None: ...

    def setXLink(self, view: Any) -> None: ...

    def setYLink(self, view: Any) -> None: ...

    def viewRect(self) -> Any: ...  # returns QtCore.QRectF

    def setMouseEnabled(self, x: bool = True, y: bool = True) -> None: ...

    def enableAutoRange(
        self,
        axis: Any = None,
        enable: bool = True,
        x: Optional[bool] = None,
        y: Optional[bool] = None,
    ) -> None: ...

    def disableAutoRange(self, axis: Any = None) -> None: ...

    def setAspectLocked(
        self,
        lock: bool = True,
        ratio: Optional[float] = None,
    ) -> None: ...

    def setLimits(self, **kargs: Any) -> None: ...

    def register(self, name: str) -> None: ...

    def unregister(self) -> None: ...
