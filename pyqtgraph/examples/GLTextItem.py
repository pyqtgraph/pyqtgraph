"""
Simple examples demonstrating the use of GLTextItem.

"""

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import mkQApp, QtCore


def add_origin(view_widget, loc, length=0.5, color='y', width=1):
    pos = [
        [loc[0], loc[1], loc[2]],
        [loc[0] + length, loc[1], loc[2]],
        [loc[0], loc[1], loc[2]],
        [loc[0], loc[1] + length, loc[2]],
        [loc[0], loc[1], loc[2]],
        [loc[0], loc[1], loc[2] + length],
    ]
    line = gl.GLLinePlotItem(
        pos=pos,
        color=pg.mkColor(color),
        width=width,
        mode='lines',
        antialias=True
    )
    view_widget.addItem(line)


def add_origin_text(view_widget, pos, alignment, text):
    txt_item = gl.GLTextItem()
    txt_item.setData(
        pos=pos,
        color=(127, 255, 127, 255),
        text=text,
        alignment=alignment
    )
    view_widget.addItem(txt_item)
    add_origin(view_widget, loc=pos)


app = mkQApp("GLTextItem Example")

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

add_origin_text(gvw, (-3.0, 2.0, 0.0), af.AlignLeft | af.AlignBottom, 'LB')
add_origin_text(gvw, (-1.0, 2.0, 0.0), af.AlignLeft | af.AlignTop, 'LT')
add_origin_text(gvw, (+1.0, 2.0, 0.0), af.AlignRight | af.AlignBottom, 'RB')
add_origin_text(gvw, (+3.0, 2.0, 0.0), af.AlignRight | af.AlignTop, 'RT')

add_origin_text(gvw, (-3.0, -2.0, 0.0), af.AlignHCenter | af.AlignBottom, 'CB')
add_origin_text(gvw, (-1.0, -2.0, 0.0), af.AlignHCenter | af.AlignTop, 'CT')

add_origin_text(gvw, (+1.0, -2.0, 0.0), af.AlignLeft | af.AlignVCenter, 'LC')
add_origin_text(gvw, (+3.0, -2.0, 0.0), af.AlignRight | af.AlignVCenter, 'RC')

add_origin_text(gvw, (2.0, +0.0, 0.0), af.AlignHCenter | af.AlignVCenter, 'CC')


if __name__ == '__main__':
  pg.exec()
