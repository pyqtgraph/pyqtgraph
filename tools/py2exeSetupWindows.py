"""
Example distutils setup script for packaging a program with
pyqtgraph and py2exe. See the packaging tutorial at
http://luke.campagnola.me/code/pyqtgraph for more information.
"""

from distutils.core import setup
from glob import glob
import py2exe
import sys

## This path must contain msvcm90.dll, msvcp90.dll, msvcr90.dll, and Microsoft.VC90.CRT.manifest
## (see http://www.py2exe.org/index.cgi/Tutorial)
dllpath = r'C:\Windows\WinSxS\x86_Microsoft.VC90.CRT...'

sys.path.append(dllpath)
data_files = [
    ## Instruct setup to copy the needed DLL files into the build directory
    ("Microsoft.VC90.CRT", glob(dllpath + r'\*.*')),
]

setup(
    data_files=data_files,
    windows=['main.py'] ,
    options={"py2exe": {"excludes":["Tkconstants", "Tkinter", "tcl"]}}
)
