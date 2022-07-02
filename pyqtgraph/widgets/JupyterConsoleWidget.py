from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.manager import QtKernelManager

# The ID of an installed kernel, e.g. 'bash' or 'ir'.
from pyqtgraph.Qt import QtWidgets

USE_KERNEL = "python3"

# This function was copied from the qtconsole embedding example code
# https://github.com/jupyter/qtconsole/blob/b4e08f763ef1334d3560d8dac1d7f9095859545a/examples/embed_qtconsole.py#L19
def make_jupyter_widget_with_kernel(dark_mode: bool = True) -> RichJupyterWidget:
    """Start a kernel, connect to it, and create a RichJupyterWidget to use it"""
    kernel_manager = QtKernelManager(kernel_name=USE_KERNEL)
    kernel_manager.start_kernel()

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    jupyter_widget = RichJupyterWidget()
    if dark_mode:
        jupyter_widget.set_default_style(
            "linux"
        )  # Dark bg color.... only key to get it...

    jupyter_widget.kernel_manager = kernel_manager
    jupyter_widget.kernel_client = kernel_client
    return jupyter_widget


class JupyterConsoleWidget(QtWidgets.QTabWidget):
    def __init__(self, *, parent=None, dark_mode: bool = True):
        super().__init__(parent=parent)

        self.rich_jupyter_widget = make_jupyter_widget_with_kernel(dark_mode=dark_mode)
        self.addTab(self.rich_jupyter_widget, "console")

    def execute_command(self, command_string: str = None) -> bool:
        return self.rich_jupyter_widget.execute(command_string)

    def close(self):
        self.rich_jupyter_widget.kernel_client.stop_channels()
        self.rich_jupyter_widget.kernel_manager.shutdown_kernel()
        super().close()

    def poll(self):
        pass
