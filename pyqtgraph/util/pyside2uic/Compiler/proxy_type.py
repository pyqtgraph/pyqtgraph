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

from pyside2uic.Compiler.misc import Literal, moduleMember


class ProxyType(type):
    def __init__(*args):
        type.__init__(*args)
        for cls in args[0].__dict__.values():
            if type(cls) is ProxyType:
                cls.module = args[0].__name__

        if not hasattr(args[0], "module"):
            args[0].module = ""

    def __getattribute__(cls, name):
        try:
            return type.__getattribute__(cls, name)
        except AttributeError:
            # Handle internal (ie. non-PySide) attributes as normal.
            if name == "module":
                raise

            # Avoid a circular import.
            from pyside2uic.Compiler.qtproxies import LiteralProxyClass

            return type(name, (LiteralProxyClass, ),
                        {"module": moduleMember(type.__getattribute__(cls, "module"),
                                                type.__getattribute__(cls, "__name__"))})

    def __str__(cls):
        return moduleMember(type.__getattribute__(cls, "module"),
                            type.__getattribute__(cls, "__name__"))

    def __or__(self, r_op):
        return Literal("%s|%s" % (self, r_op))

    def __eq__(self, other):
        return str(self) == str(other)
