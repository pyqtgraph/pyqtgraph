import os, sys
## Search the package tree for all .ui files, compile each to
## a .py for pyqt and pyside

pyqtuic = 'pyuic4'
pysideuic = 'pyside-uic'
pyqt5uic = 'pyuic5'

for path, sd, files in os.walk('.'):
    for f in files:
        base, ext = os.path.splitext(f)
        if ext != '.ui':
            continue
        ui = os.path.join(path, f)

        py = os.path.join(path, base + '_pyqt.py')
        if not os.path.exists(py) or os.stat(ui).st_mtime > os.stat(py).st_mtime:
            os.system('%s %s > %s' % (pyqtuic, ui, py))
            print(py)

        py = os.path.join(path, base + '_pyside.py')
        if not os.path.exists(py) or os.stat(ui).st_mtime > os.stat(py).st_mtime:
            os.system('%s %s > %s' % (pysideuic, ui, py))
            print(py)

        py = os.path.join(path, base + '_pyqt5.py')
        if not os.path.exists(py) or os.stat(ui).st_mtime > os.stat(py).st_mtime:
            os.system('%s %s > %s' % (pyqt5uic, ui, py))
            print(py)

