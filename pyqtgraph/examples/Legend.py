"""
Demonstrates basic use of LegendItem
"""

from pyqtgraph.Qt.QtWidgets import QInputDialog

import numpy as np

import pyqtgraph as pg

win = pg.plot()
win.setWindowTitle('pyqtgraph example: BarGraphItem')

# # option1: only for .plot(), following c1,c2 for example-----------------------
# win.addLegend(frame=False, colCount=2)

# bar graph
x = np.arange(10)
y = np.sin(x+2) * 3
bg1 = pg.BarGraphItem(x=x, height=y, width=0.3, brush='b', pen='w', name='bar')
win.addItem(bg1)

# curve
c1 = win.plot([np.random.randint(0,8) for i in range(10)], pen='r', symbol='t', symbolPen='r', symbolBrush='g', name='curve1')
c2 = win.plot([2,1,4,3,1,3,2,4,3,2], pen='g', fillLevel=0, fillBrush=(255,255,255,30), name='curve2')

# scatter plot
s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120), name='scatter')
spots = [{'pos': [i, np.random.randint(-3, 3)], 'data': 1} for i in range(10)]
s1.addPoints(spots)
win.addItem(s1)

# # option2: generic method------------------------------------------------
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win.graphicsItem())
legend.addItem(bg1, 'bar')
legend.addItem(c1, 'curve1')
legend.addItem(c2, 'curve2')
legend.addItem(s1, 'scatter')


def legendDoubleClicked(legend, event):
    # update legend font size and redraw
    current_font_size = int(legend.labelTextSize().replace('pt', ''))

    dialog = QInputDialog()
    dialog.setWindowTitle("Input Dialog")
    dialog.setInputMode(QInputDialog.InputMode.IntInput)
    dialog.setLabelText('Enter Label Font Size:')
    dialog.setIntValue(current_font_size)

    if dialog.exec() == QInputDialog.DialogCode.Accepted:
        font_size = dialog.intValue()
        legend.setLabelTextSize('%dpt' % font_size)
        legend_items = legend.items.copy()
        legend.clear()
        # re-add items to update labels and redraw
        for sample, label in legend_items:
            legend.addItem(sample.item, label.text)

    event.accept()


def legendSampleClicked(plot_data_item):
    # indicate plot data item visibility
    if plot_data_item.isVisible():
        print('"%s" is visible' % plot_data_item.name())
    else:
        print('"%s" is not visible' % plot_data_item.name())


legend.sigDoubleClicked.connect(legendDoubleClicked)
legend.sigSampleClicked.connect(legendSampleClicked)


if __name__ == '__main__':
    pg.exec()
