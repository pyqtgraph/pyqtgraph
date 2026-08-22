"""
Simple examples demonstrating the use of GLTextItem.

"""

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

def get_axes_lines(loc, length=0.5):
    return [
        [loc[0], loc[1], loc[2]],
        [loc[0] + length, loc[1], loc[2]],
        [loc[0], loc[1], loc[2]],
        [loc[0], loc[1] + length, loc[2]],
        [loc[0], loc[1], loc[2]],
        [loc[0], loc[1], loc[2] + length],
    ]

pg.mkQApp("GLTextItem Example")

gvw = gl.GLViewWidget()
gvw.show()
gvw.setWindowTitle('pyqtgraph example: GLTextItem')

griditem = gl.GLGridItem()
griditem.setSize(10, 10)
griditem.setSpacing(1, 1)
gvw.addItem(griditem)

axisitem = gl.GLAxisItem()
gvw.addItem(axisitem)

txtitem1 = gl.GLTextItem(pos=(0.0, 0.0, 0.0), text='text1')
gvw.addItem(txtitem1)

txtitem2 = gl.GLTextItem(pos=(1.0, -1.0, 2.0), color=(127, 255, 127, 255), text='text2')
gvw.addItem(txtitem2)

af = QtCore.Qt.AlignmentFlag
hlut = dict(L=af.AlignLeft, C=af.AlignHCenter, R=af.AlignRight)
vlut = dict(B=af.AlignBottom, C=af.AlignVCenter, T=af.AlignTop)

params = [
    ((-3.0, 2.0, 0.0), 'LB'),
    ((-1.0, 2.0, 0.0), 'LT'),
    ((+1.0, 2.0, 0.0), 'RB'),
    ((+3.0, 2.0, 0.0), 'RT'),

    ((-3.0, -2.0, 0.0), 'CB'),
    ((-1.0, -2.0, 0.0), 'CT'),

    ((+1.0, -2.0, 0.0), 'LC'),
    ((+3.0, -2.0, 0.0), 'RC'),

    ((2.0, +0.0, 0.0), 'CC'),
]

lines = []
words = []
for pos, text in params:
    lines.extend(get_axes_lines(pos))
    words.append(dict(text=text, pos=pos, alignment=hlut[text[0]] | vlut[text[1]]))

axes_lines = gl.GLLinePlotItem(
    pos=lines,
    color=pg.mkColor('y'),
    width=1.0,
    mode='lines',
    antialias=True
)
gvw.addItem(axes_lines)

axes_words = gl.GLTextItem(color=(127, 255, 127, 255), items=words)
gvw.addItem(axes_words)

if __name__ == '__main__':
    pg.exec()
