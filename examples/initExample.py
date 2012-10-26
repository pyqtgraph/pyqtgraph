## make this version of pyqtgraph importable before any others
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if 'pyside' in sys.argv:  ## should force example to use PySide instead of PyQt
    import PySide
elif 'pyqt' in sys.argv: 
    import PyQt4
