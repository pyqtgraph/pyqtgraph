"""
Simple demonstration of TreeWidget, which is an extension of QTreeWidget
that allows widgets to be added and dragged within the tree more easily.
"""


import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("TreeWidget Example")

w = pg.TreeWidget()
w.setColumnCount(2)
w.show()
w.setWindowTitle('pyqtgraph example: TreeWidget')

i1  = QtWidgets.QTreeWidgetItem(["Item 1"])
i11  = QtWidgets.QTreeWidgetItem(["Item 1.1"])
i12  = QtWidgets.QTreeWidgetItem(["Item 1.2"])
i2  = QtWidgets.QTreeWidgetItem(["Item 2"])
i21  = QtWidgets.QTreeWidgetItem(["Item 2.1"])
i211  = pg.TreeWidgetItem(["Item 2.1.1"])
i212  = pg.TreeWidgetItem(["Item 2.1.2"])
i22  = pg.TreeWidgetItem(["Item 2.2"])
i3  = pg.TreeWidgetItem(["Item 3"])
i4  = pg.TreeWidgetItem(["Item 4"])
i5  = pg.TreeWidgetItem(["Item 5"])
b5 = QtWidgets.QPushButton('Button')
i5.setWidget(1, b5)



w.addTopLevelItem(i1)
w.addTopLevelItem(i2)
w.addTopLevelItem(i3)
w.addTopLevelItem(i4)
w.addTopLevelItem(i5)
i1.addChild(i11)
i1.addChild(i12)
i2.addChild(i21)
i21.addChild(i211)
i21.addChild(i212)
i2.addChild(i22)

b1 = QtWidgets.QPushButton("Button")
w.setItemWidget(i1, 1, b1)

if __name__ == '__main__':
    pg.exec()
