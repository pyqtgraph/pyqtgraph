import os
import re

from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem


def _set_filepicker_kwargs(fileDlg, **kwargs):
  """Applies a dict of enum/flag kwarg opts to a file dialog"""
  NO_MATCH = object()

  for kk, vv in kwargs.items():
    # Convert string or list representations into true flags
    # 'fileMode' -> 'FileMode'
    formattedName = kk[0].upper() + kk[1:]
    # Edge case: "Options" has enum "Option"
    if formattedName == 'Options':
      enumCls = fileDlg.Option
    else:
      enumCls = getattr(fileDlg, formattedName, NO_MATCH)
    setFunc = getattr(fileDlg, f'set{formattedName}', NO_MATCH)
    if enumCls is NO_MATCH or setFunc is NO_MATCH:
      continue
    if enumCls is fileDlg.Option:
      builder = fileDlg.Option(0)
      # This is the only flag enum, all others can only take one value
      if isinstance(vv, str): vv = [vv]
      for flag in vv:
        curVal = getattr(enumCls, flag)
        builder |= curVal
      # Some Qt implementations turn into ints by this point
      outEnum = enumCls(builder)
    else:
      outEnum = getattr(enumCls, vv)
    setFunc(outEnum)


def popupFilePicker(parent=None, windowTitle='', nameFilter='', directory=None, selectFile=None, relativeTo=None, **kwargs):
    """
    Thin wrapper around Qt file picker dialog. Used internally so all options are consistent
    among all requests for external file information

    ============== ========================================================
    **Arguments:**
    parent         Dialog parent
    windowTitle    Title of dialog window
    nameFilter     File filter as required by the Qt dialog
    directory      Where in the file system to open this dialog
    selectFile     File to preselect
    relativeTo     Parent directory that, if provided, will be removed from the prefix of all returned paths. So,
                   if '/my/text/file.txt' was selected, and `relativeTo='/my/text/'`, the return value would be
                   'file.txt'. This uses os.path.relpath under the hood, so expect that behavior.
    kwargs         Any enum value accepted by a QFileDialog and its value. Values can be a string or list of strings,
                   i.e. fileMode='AnyFile', options=['ShowDirsOnly', 'DontResolveSymlinks'], acceptMode='AcceptSave'
    ============== ========================================================

    """
    fileDlg = QtWidgets.QFileDialog(parent)
    _set_filepicker_kwargs(fileDlg, **kwargs)

    fileDlg.setModal(True)
    if directory is not None:
        fileDlg.setDirectory(directory)
    fileDlg.setNameFilter(nameFilter)
    if selectFile is not None:
        fileDlg.selectFile(selectFile)

    fileDlg.setWindowTitle(windowTitle)

    if fileDlg.exec():
        # Append filter type
        singleExtReg = r'(\.\w+)'
        # Extensions of type 'myfile.ext.is.multi.part' need to capture repeating pattern of singleExt
        suffMatch = re.search(rf'({singleExtReg}+)', fileDlg.selectedNameFilter())
        if suffMatch:
            # Strip leading '.' if it exists
            ext = suffMatch.group(1)
            if ext.startswith('.'):
                ext = ext[1:]
            fileDlg.setDefaultSuffix(ext)
        fList = fileDlg.selectedFiles()
    else:
        fList = []
    if relativeTo is not None:
        fList = [os.path.relpath(file, relativeTo) for file in fList]
    # Make consistent to os flavor
    fList = [os.path.normpath(file) for file in fList]
    if fileDlg.fileMode() == fileDlg.FileMode.ExistingFiles:
        return fList
    elif len(fList) > 0:
        return fList[0]
    else:
        return None


class FileParameterItem(StrParameterItem):
    def __init__(self, param, depth):
        self._value = None
        super().__init__(param, depth)

        button = QtWidgets.QPushButton('...')
        button.setFixedWidth(25)
        button.setContentsMargins(0, 0, 0, 0)
        button.clicked.connect(self._retrieveFileSelection_gui)
        self.layoutWidget.layout().insertWidget(2, button)
        self.displayLabel.resizeEvent = self._newResizeEvent
        # self.layoutWidget.layout().insertWidget(3, self.defaultBtn)

    def makeWidget(self):
        w = super().makeWidget()
        w.setValue = self.setValue
        w.value = self.value
        # Doesn't make much sense to have a 'changing' signal since filepaths should be complete before value
        # is emitted
        delattr(w, 'sigChanging')
        return w

    def _newResizeEvent(self, ev):
        ret = type(self.displayLabel).resizeEvent(self.displayLabel, ev)
        self.updateDisplayLabel()
        return ret

    def setValue(self, value):
        self._value = value
        self.widget.setText(str(value))

    def value(self):
        return self._value

    def _retrieveFileSelection_gui(self):
        curVal = self.param.value()
        if isinstance(curVal, list) and len(curVal):
            # All files should be from the same directory, in principle
            # Since no mechanism exists for preselecting multiple, the most sensible
            # thing is to select nothing in the preview dialog
            curVal = curVal[0]
            if os.path.isfile(curVal):
                curVal = os.path.dirname(curVal)
        opts = self.param.opts.copy()
        useDir = curVal or opts.get('directory') or os.getcwd()
        startDir = os.path.abspath(useDir)
        if os.path.isfile(startDir):
            opts['selectFile'] = os.path.basename(startDir)
            startDir = os.path.dirname(startDir)
        if os.path.exists(startDir):
            opts['directory'] = startDir
        if opts.get('windowTitle') is None:
            opts['windowTitle'] = self.param.title()

        fname = popupFilePicker(None, **opts)
        if not fname:
            return
        self.param.setValue(fname)

    def updateDefaultBtn(self):
        # Override since a readonly label should still allow reverting to default
        ## enable/disable default btn
        self.defaultBtn.setEnabled(
            not self.param.valueIsDefault() and self.param.opts['enabled'])

        # hide / show
        self.defaultBtn.setVisible(self.param.hasDefault())

    def updateDisplayLabel(self, value=None):
        lbl = self.displayLabel
        if value is None:
            value = self.param.value()
        value = str(value)
        font = lbl.font()
        metrics = QtGui.QFontMetricsF(font)
        value = metrics.elidedText(value, QtCore.Qt.TextElideMode.ElideLeft, lbl.width()-5)
        return super().updateDisplayLabel(value)


class FileParameter(Parameter):
    """
    Interfaces with the myriad of file options available from a QFileDialog.

    Note that the output can either be a single file string or list of files, depending on whether
    `fileMode='ExistingFiles'` is specified.

    Note that in all cases, absolute file paths are returned unless `relativeTo` is specified as
    elaborated below.

    ============== ========================================================
    **Options:**
    parent         Dialog parent
    winTitle       Title of dialog window
    nameFilter     File filter as required by the Qt dialog
    directory      Where in the file system to open this dialog
    selectFile     File to preselect
    relativeTo     Parent directory that, if provided, will be removed from the prefix of all returned paths. So,
                   if '/my/text/file.txt' was selected, and `relativeTo='my/text/'`, the return value would be
                   'file.txt'. This uses os.path.relpath under the hood, so expect that behavior.
    kwargs         Any enum value accepted by a QFileDialog and its value. Values can be a string or list of strings,
                   i.e. fileMode='AnyFile', options=['ShowDirsOnly', 'DontResolveSymlinks']
    ============== ========================================================
    """
    itemClass = FileParameterItem

    def __init__(self, **opts):
        opts.setdefault('readonly', True)
        super().__init__(**opts)
