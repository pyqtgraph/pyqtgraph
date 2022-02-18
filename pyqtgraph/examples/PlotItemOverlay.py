import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.graphicsItems.AxisItem import AxisItem
# from pyqtgraph.graphicsItems.PlotItem import PlotItem
from pyqtgraph.widgets._plotitemoverlay import PlotItemOverlay
from pyqtgraph.Qt.QtCore import pyqtRemoveInputHook

app = pg.mkQApp()
pyqtRemoveInputHook()
mw = QtWidgets.QMainWindow()
mw.resize(800, 800)
pg.setConfigOption("background", "black")
pg.setConfigOption("foreground", "white")


def mk_axes(names: list[str] = ['right', 'bottom']):
    return {name: AxisItem(orientation=name) for name in names}


pw = pg.PlotWidget(
    axisItems=mk_axes(),
    # XXX: this is needed here (and below) to avoid
    # "default visible axes" from being displayed which
    # not only hurts the overlayed axes performance but
    # also results in placement of undesired axes..
    default_axes=[],
)
mw.setCentralWidget(pw)
mw.show()
pw.addLegend(offset=(0, 0))
pw.setTitle("PlotItem Overlay Example")

plot1 = pw.plotItem
overlay = PlotItemOverlay(plot1)

# for prints inside `ViewBox`
plot1.vb.name = 'root'
# plot1.setTitle(f'Dataset root')

# XXX: this works now on each individual plot.
# plot1.enableAutoRange(axis='y')
# plot1.setAutoVisible(y=True)

f = 2
data1 = np.sin(np.linspace(0, f * np.pi, num=1000))
data2 = data1 * 2
data3 = np.sin(np.linspace(0, f*2 * np.pi, num=500)) * 3
data4 = np.sin(np.linspace(0, f*4 * np.pi, num=500)) * 4

# Use std plot api for "top" plotitem
plot1.plot().setData(data1)
plot1.update()
plot1.show()

for i, (plot, data) in enumerate((
    (pg.PlotItem(
        axisItems=mk_axes(),
        parent=plot1,
        default_axes=[],),
     data2),
    (pg.PlotItem(
        axisItems=mk_axes(),
        parent=plot1,
        default_axes=[],),
     data3),
    (pg.PlotItem(
        axisItems=mk_axes(),
        parent=plot1,
        default_axes=[],),
     data4),
)):
    # TODO: get title stacking in the layout working

    # NOTE: if we add this it causes mouse wheel zoom
    # to be offset in a wonky way..
    # I'm hoping re-stacking labels in the top level layout
    # will fix this.
    # plot.setTitle(f'Dataset {i}')

    plot.hideButtons()  # see features notes in overlay module
    plot.plot().setData(data)

    # XXX: oh look, now this works because we have separate viewboxex B)
    plot.enableAutoRange(axis='y')
    plot.setAutoVisible(y=True)

    # XXX: no extra APIs needed ;)
    overlay.add_plotitem(plot)

    # XXX: for internal viewbox prints/debug
    plot.vb.name = i

if __name__ == '__main__':
    pg.exec()
