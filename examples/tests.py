
from .__main__ import buildFileList, testFile, sys, examples

def test_pyside():
    files = buildFileList(examples)
    for f in files:
        yield testFile, f[0], f[1], sys.executable, 'PySide'
        # testFile(f[0], f[1], sys.executable, 'PySide')
