import sys

from ..Qt import QtWidgets

__all__ = ['FileDialog']

class FileDialog(QtWidgets.QFileDialog):
    ## Compatibility fix for OSX:
    ## For some reason the native dialog doesn't show up when you set AcceptMode to AcceptSave on OS X, so we don't use the native dialog    
    
    def __init__(self, *args):
        QtWidgets.QFileDialog.__init__(self, *args)
        
        if sys.platform == 'darwin': 
            self.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
