import pyqtgraph as pg
pg.mkQApp()
pw = pg.PlotWidget()

text_item = pg.TextItem("Any text", ensureInBounds=True)
pw.addItem(text_item)
text_item = pg.TextItem("Some other text", ensureInBounds=True)
text_item.setPos(10, 10)
pw.addItem(text_item)

pw.show()
pg.exec()