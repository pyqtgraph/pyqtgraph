"""
 This function was copied from the qtconsole embedding example code
 https://github.com/jupyter/qtconsole/blob/b4e08f763ef1334d3560d8dac1d7f9095859545a/examples/embed_qtconsole.py#L19

 This widget was adapted from the implementation from - https://gitlab.com/yaq/yaqc-qtpy/-/blob/main/yaqc_qtpy/_main_widget.py

"""
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.widgets.JupyterConsoleWidget import JupyterConsoleWidget

app = pg.mkQApp("ImageView Example")

## Create window with ImageView widget
main_window_widget = QtWidgets.QMainWindow()
main_window_widget.resize(800, 800)

## create JupyterConsoleWidget (and connect to kernel)
jupyter_console_widget = JupyterConsoleWidget()
main_window_widget.setCentralWidget(jupyter_console_widget)
main_window_widget.setWindowTitle("pyqtgraph example: RichJupyterConsole")
main_window_widget.show()

jupyter_console_widget.execute_command("import numpy as np ")
jupyter_console_widget.execute_command("x = np.arange(0,2*np.pi,.1)")
jupyter_console_widget.execute_command("print(x.shape)")
# jupyter_console_widget.execute_command("!pip install matplotlib  ") #uncomment to install `matplotlib` to this environment
jupyter_console_widget.execute_command("import matplotlib.pyplot as plt")
jupyter_console_widget.execute_command("%matplotlib inline")
jupyter_console_widget.execute_command("plt.plot(np.sin(x))")

jupyter_console_widget.execute_command('print("Good Luck; Have Fun! 😄✨")')
jupyter_console_widget.execute_command("")


if __name__ == "__main__":
    pg.exec()
