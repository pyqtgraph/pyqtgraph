rem
rem   This is a simple windows batch file containing the commands needed to package
rem   a program with pyqtgraph and py2exe. See the packaging tutorial at
rem   http://luke.campagnola.me/code/pyqtgraph for more information.
rem 

rmdir /S /Q dist
rmdir /S /Q build
python .\py2exeSetupWindows.py py2exe --includes sip
pause
