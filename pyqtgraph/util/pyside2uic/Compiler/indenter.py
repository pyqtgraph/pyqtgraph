# This file is part of the PySide project.
#
# Copyright (C) 2009 Nokia Corporation and/or its subsidiary(-ies).
# Copyright (C) 2009 Riverbank Computing Limited.
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

indentwidth = 4

_indenter = None

class _IndentedCodeWriter(object):
    def __init__(self, output):
        self.level = 0
        self.output = output

    def indent(self):
        self.level += 1

    def dedent(self):
        self.level -= 1

    def write(self, line):
        if line.strip():
            if indentwidth > 0:
                indent = " " * indentwidth
                line = line.replace("\t", indent)
            else:
                indent = "\t"

            self.output.write("%s%s\n" % (indent * self.level, line))
        else:
            self.output.write("\n")


def createCodeIndenter(output):
    global _indenter
    _indenter = _IndentedCodeWriter(output)

def getIndenter():
    return _indenter

def write_code(string):
    _indenter.write(string)
