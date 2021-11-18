"""
Simple examples demonstrating the use of GLTextItem.

"""

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import mkQApp

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

txtitem2 = gl.GLTextItem()
txtitem2.setData(pos=(1.0, -1.0, 2.0), color=(127, 255, 127, 255), text='text2')
gvw.addItem(txtitem2)

if __name__ == '__main__':
  pg.exec()
