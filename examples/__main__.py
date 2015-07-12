import sys, os
import pyqtgraph as pg


if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"

from .utils import buildFileList, testFile, run, path, examples

if __name__ == '__main__':

    args = sys.argv[1:]
        
    if '--test' in args:
        # get rid of orphaned cache files first
        pg.renamePyc(path)
        
        files = buildFileList(examples)
        if '--pyside' in args:
            lib = 'PySide'
        elif '--pyqt' in args or '--pyqt4' in args:
            lib = 'PyQt4'
        elif '--pyqt5' in args:
            lib = 'PyQt5'
        else:
            lib = ''
            
        exe = sys.executable
        print("Running tests:", lib, sys.executable)
        for f in files:
            testFile(f[0], f[1], exe, lib)
    else: 
        run()
