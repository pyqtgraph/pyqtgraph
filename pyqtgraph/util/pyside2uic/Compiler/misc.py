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


from pyside2uic.Compiler.indenter import write_code


def write_import(module_name, from_imports):
    if from_imports:
        write_code("from . import %s" % module_name)
    else:
        write_code("import %s" % module_name)


def moduleMember(module, name):
    if module:
        return "%s.%s" % (module, name)

    return name


class Literal(object):
    """Literal(string) -> new literal

    string will not be quoted when put into an argument list"""
    def __init__(self, string):
        self.string = string

    def __str__(self):
        return self.string

    def __or__(self, r_op):
        return Literal("%s|%s" % (self, r_op))
