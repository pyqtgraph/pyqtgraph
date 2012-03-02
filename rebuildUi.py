import os, sys
## run "python rebuildUi.py pyside" to rebuild all ui files for pyside

uic = 'pyuic4'
if len(sys.argv) > 1 and sys.argv[1] == 'pyside':
    uic = 'pyside-uic'

for path, sd, files in os.walk('.'):
    for f in files:
        base, ext = os.path.splitext(f)
        if ext != '.ui':
            continue
        ui = os.path.join(path, f)
        py = os.path.join(path, base + '.py')
        os.system('%s %s > %s' % (uic, ui, py))
        print py
