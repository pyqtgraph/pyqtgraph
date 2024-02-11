import subprocess
import sys


def test_canvas_manager_singleton_crash() -> None:
    """Regression test for #2838."""
    proc = subprocess.run([sys.executable, "-c", "import pyqtgraph.canvas"])
    assert proc.returncode == 0
