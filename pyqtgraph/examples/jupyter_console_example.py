import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

try:
    from qtconsole import inprocess
except:
    print("The example in `jupyter_console_example.py` requires `qtconsole` to run. Install with `pip install qtconsole` or equivalent.")



class JupyterConsoleWidget(inprocess.QtInProcessRichJupyterWidget):
    def __init__(self):
        super().__init__()

        self.kernel_manager = inprocess.QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

    def shutdown_kernel(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dark_mode=True):
        super().__init__()
        central_widget = QtWidgets.QWidget()
        vbox_widget = QtWidgets.QVBoxLayout()
        central_widget.setLayout(vbox_widget)
        self.plot_widget = pg.PlotWidget()
        self.jupyter_console_widget = JupyterConsoleWidget()
        vbox_widget.addWidget(self.plot_widget)
        vbox_widget.addWidget(self.jupyter_console_widget)
        self.setCentralWidget(central_widget)

        app = QtWidgets.QApplication.instance()
        app.aboutToQuit.connect(self.jupyter_console_widget.shutdown_kernel)

        kernel = self.jupyter_console_widget.kernel_manager.kernel
        kernel.shell.push(dict(np=np, pw=self.plot_widget))

        #set dark mode
        if dark_mode:
            self.jupyter_console_widget.set_default_style("linux")  # Dark bg color.... only key to get it...



if __name__ == '__main__':
    pg.mkQApp()
    main = MainWindow(dark_mode=True)
    main.show()
    main.jupyter_console_widget.execute("print(\"hello world :D \")")

    #plot a sine/cosine waves by printing to console (as if you were typing the commands in manually)
    main.jupyter_console_widget.execute("x = np.arange(0, 3 * np.pi, .1)")
    main.jupyter_console_widget.execute("pw.plotItem.plot(np.sin(x), pen='r')")
    main.jupyter_console_widget.execute("pw.plotItem.plot(np.cos(x), pen='cyan', symbol='o',symbolPen='m', symbolBrush=(0,0,255))")
    main.jupyter_console_widget.execute("whos")
    main.jupyter_console_widget.execute("")

    pg.exec()