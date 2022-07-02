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
input_variables_namespace_dict = {"i_put_this_variable_in_at_startup": True}
jupyter_console_widget = JupyterConsoleWidget(namespace=input_variables_namespace_dict)
main_window_widget.setCentralWidget(jupyter_console_widget)
main_window_widget.setWindowTitle("pyqtgraph example: RichJupyterConsole")
main_window_widget.show()

jupyter_console_widget.execute_command("i_put_this_variable_in_at_startup")
jupyter_console_widget.execute_command("import numpy as np ")
jupyter_console_widget.execute_command("x = np.arange(0,2*np.pi,.1)")
jupyter_console_widget.execute_command("print(x.shape)")
# jupyter_console_widget.execute_command("!pip install matplotlib  ") #uncomment to install `matplotlib` to this environment
jupyter_console_widget.execute_command("import matplotlib.pyplot as plt")
jupyter_console_widget.execute_command("%matplotlib inline")
jupyter_console_widget.execute_command("plt.plot(np.sin(x))")
jupyter_console_widget.execute_command("print('----')")

## Push variables into Console namespace with `jupyter_console_widget.push_variables(variables_dict)`


variables_dict = {"this_is_an_int": 9, "this_is_a_string": "hello :D"}

jupyter_console_widget.push_variables(variables_dict)

jupyter_console_widget.execute_command("this_is_an_int")
jupyter_console_widget.execute_command("this_is_a_string")

jupyter_console_widget.execute_command("print('----')")

jupyter_console_widget.execute_command('print("Good Luck; Have Fun! 😄✨")')
jupyter_console_widget.execute_command("")


if __name__ == "__main__":
    pg.exec()
