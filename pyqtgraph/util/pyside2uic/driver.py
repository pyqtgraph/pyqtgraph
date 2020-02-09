# This file is part of the PySide project.
#
# Copyright (C) 2009-2011 Nokia Corporation and/or its subsidiary(-ies).
# Copyright (C) 2010 Riverbank Computing Limited.
# Copyright (C) 2009 Torsten Marek
#
# Contact: PySide team <pyside@openbossa.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 2 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
# 02110-1301 USA


import sys
import logging

from . import compileUi


class Driver(object):
    """ This encapsulates access to the pyuic functionality so that it can be
    called by code that is Python v2/v3 specific.
    """

    LOGGER_NAME = 'PySide2.uic'

    def __init__(self, opts, ui_file):
        """ Initialise the object.  opts is the parsed options.  ui_file is the
        name of the .ui file.
        """

        if opts.debug:
            logger = logging.getLogger(self.LOGGER_NAME)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        self._opts = opts
        self._ui_file = ui_file

    def invoke(self):
        """ Invoke the action as specified by the parsed options.  Returns 0 if
        there was no error.
        """

        if self._opts.preview:
            return self._preview()

        self._generate()

        return 0

    def _preview(self):
        """ Preview the .ui file.  Return the exit status to be passed back to
        the parent process.
        """

        from PySide2 import QtUiTools
        from PySide2 import QtGui
        from PySide2 import QtWidgets

        app = QtWidgets.QApplication([self._ui_file])
        widget = QtUiTools.QUiLoader().load(self._ui_file)
        widget.show()

        return app.exec_()

    def _generate(self):
        """ Generate the Python code. """

        if sys.hexversion >= 0x03000000:
            if self._opts.output == '-':
                from io import TextIOWrapper

                pyfile = TextIOWrapper(sys.stdout.buffer, encoding='utf8')
            else:
                pyfile = open(self._opts.output, 'wt', encoding='utf8')
        else:
            if self._opts.output == '-':
                pyfile = sys.stdout
            else:
                pyfile = open(self._opts.output, 'wt')

        compileUi(self._ui_file, pyfile, self._opts.execute, self._opts.indent, self._opts.from_imports)

    def on_IOError(self, e):
        """ Handle an IOError exception. """

        sys.stderr.write("Error: %s: \"%s\"\n" % (e.strerror, e.filename))

    def on_SyntaxError(self, e):
        """ Handle a SyntaxError exception. """

        sys.stderr.write("Error in input file: %s\n" % e)

    def on_NoSuchWidgetError(self, e):
        """ Handle a NoSuchWidgetError exception. """

        if e.args[0].startswith("Q3"):
            sys.stderr.write("Error: Q3Support widgets are not supported by PySide2.\n")
        else:
            sys.stderr.write(str(e) + "\n")

    def on_Exception(self, e):
        """ Handle a generic exception. """

        if logging.getLogger(self.LOGGER_NAME).level == logging.DEBUG:
            import traceback

            traceback.print_exception(*sys.exc_info())
        else:
            from PySide2 import QtCore

            sys.stderr.write("""An unexpected error occurred.
Check that you are using the latest version of PySide2 and report the error to
http://bugs.openbossa.org, including the ui file used to trigger the error.
""")
