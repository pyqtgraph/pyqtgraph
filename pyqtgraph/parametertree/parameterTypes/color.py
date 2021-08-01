from ...widgets.ColorButton import ColorButton
from ... import functions as fn
from .basetypes import WidgetParameterItem, SimpleParameter


class ColorParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a :class:`ColorButton <pyqtgraph.ColorButton>` """
    def makeWidget(self):
        w = ColorButton()
        w.sigChanged = w.sigColorChanged
        w.sigChanging = w.sigColorChanging
        w.value = w.color
        w.setValue = w.setColor
        self.hideWidget = False
        w.setFlat(True)
        return w


class ColorParameter(SimpleParameter):
    itemClass = ColorParameterItem

    def _interpretValue(self, v):
        return fn.mkColor(v)

    def value(self):
        return fn.mkColor(super().value())

    def saveState(self, filter=None):
        state = super().saveState(filter)
        state['value'] = fn.colorTuple(self.value())
        return state
