# This Python file uses the following encoding: utf-8
"""
This example demonstrates how to makes your plotDataItem glow.
This is strongly inspired by this repository:
https://github.com/dhaitz/mplcyberpunk
"""

from pyqtgraph.Qt import QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

app = pg.mkQApp("Glow Example")

# Dedicated colors which look "good"
colors = ['#08F7FE', '#FE53BB', '#F5D300', '#00ff41', '#FF0000', '#9467bd', ]

# We create the ParameterTree
children = [
    dict(name='make_line_glow', type='bool', value=False),
    dict(name='add_underglow', type='list', limits=['None', 'Full', 'Gradient'], value='None'),
    dict(name='nb_lines', type='int', limits=[1, 6], value=1),
    dict(name='nb glow lines', type='int', limits=[0, 15], value=10),
    dict(name='alpha_start', type='int', limits=[0, 255], value=25, step=1),
    dict(name='alpha_stop', type='int', limits=[0, 255], value=25, step=1),
    dict(name='alpha_underglow', type='int', limits=[0, 255], value=25, step=1),
    dict(name='linewidth_start', type='float', limits=[0.1, 50], value=1, step=0.1),
    dict(name='linewidth_stop', type='float', limits=[0.2, 50], value=8, step=0.1),
]

params = pg.parametertree.Parameter.create(name='Parameters', type='group', children=children)
pt = pg.parametertree.ParameterTree(showHeader=False)
pt.setParameters(params)


pw2 = pg.PlotWidget()
splitter = QtWidgets.QSplitter()
splitter.addWidget(pt)
splitter.addWidget(pw2)
splitter.show()

# Add some noise on the curves
noise  = 0.1
noises: list = np.random.rand(6, 100)*noise

def update_plot():
    pw2.clear()

    nb_glow_lines   = params.child('nb glow lines').value()
    alpha_start     = params.child('alpha_start').value()
    alpha_stop      = params.child('alpha_stop').value()
    alpha_underglow = params.child('alpha_underglow').value()
    linewidth_start = params.child('linewidth_start').value()
    linewidth_stop  = params.child('linewidth_stop').value()
    nb_lines        = params.child('nb_lines').value()

    xs = []
    ys = []
    for i in range(nb_lines):
        xs.append(np.linspace(0, 2*np.pi, 100)-i)
        ys.append(np.sin(xs[-1])*xs[-1]-i/3+noises[i])

    # For each line we:
    # 1. Add a PlotDataItem with the pen and brush corresponding to the line
    #    color and the underglow
    # 2. Add nb_glow_lines PlotDatamItem with increasing width and low alpha
    #    to create the glow effect
    for color, x, y in zip(colors, xs, ys):
        pen = pg.mkPen(color=color)
        if params.child('add_underglow').value()=='Full':
            kw={'fillLevel' : 0.0,
                'fillBrush' : pg.mkBrush(color='{}{:02x}'.format(color, alpha_underglow)),
                }
        elif params.child('add_underglow').value()=='Gradient':
            grad = QtGui.QLinearGradient(x.mean(), y.min(), x.mean(), y.max())
            grad.setColorAt(0.001, pg.mkColor(color))
            grad.setColorAt(abs(y.min())/(y.max()-y.min()), pg.mkColor('{}{:02x}'.format(color, alpha_underglow)))
            grad.setColorAt(0.999, pg.mkColor(color))
            brush = QtGui.QBrush(grad)
            kw={'fillLevel' : 0.0,
                'fillBrush' : brush,
                }
        else:
            kw = {}
        pw2.addItem(pg.PlotDataItem(x, y, pen=pen, **kw))


        if params.child('make_line_glow').value():
            alphas = np.linspace(alpha_start, alpha_stop, nb_glow_lines, dtype=int)
            lws = np.linspace(linewidth_start, linewidth_stop, nb_glow_lines)

            for alpha, lw in zip(alphas, lws):

                pen = pg.mkPen(color='{}{:02x}'.format(color, alpha),
                               width=lw,
                               connect="finite")

                pw2.addItem(pg.PlotDataItem(x, y,
                                            pen=pen))

params.sigTreeStateChanged.connect(update_plot)
update_plot()


if __name__ == '__main__':
    pg.exec()
