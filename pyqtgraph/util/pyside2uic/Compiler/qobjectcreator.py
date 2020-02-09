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


import logging

try:
    set()
except NameError:
    from sets import Set as set

from .indenter import write_code
from .qtproxies import (QtWidgets, QtGui, Literal,
                                           strict_getattr)


logger = logging.getLogger(__name__)
DEBUG = logger.debug


class _QtGuiWrapper(object):
    def search(clsname):
        try:
            return strict_getattr(QtGui, clsname)
        except AttributeError:
            return None

    search = staticmethod(search)


class _QtWidgetsWrapper(object):
    def search(clsname):
        try:
            return strict_getattr(QtWidgets, clsname)
        except AttributeError:
            return None

    search = staticmethod(search)


class _ModuleWrapper(object):
    def __init__(self, name, classes):
        if "." in name:
            idx = name.rfind(".")
            self._package = name[:idx]
            self._module = name[idx + 1:]
        else:
            self._package = None
            self._module = name

        self._classes = set(classes)
        self._used = False

    def search(self, cls):
        if cls in self._classes:
            self._used = True
            return type(cls, (QtWidgets.QWidget,), {"module": self._module})
        else:
            return None

    def _writeImportCode(self):
        if self._used:
            if self._package is None:
                write_code("import %s" % self._module)
            else:
                write_code("from %s import %s" % (self._package, self._module))


class _CustomWidgetLoader(object):
    def __init__(self):
        self._widgets = {}
        self._usedWidgets = set()

    def addCustomWidget(self, widgetClass, baseClass, module):
        assert widgetClass not in self._widgets
        self._widgets[widgetClass] = (baseClass, module)


    def _resolveBaseclass(self, baseClass):
        try:
            for x in range(0, 10):
                try: return strict_getattr(QtWidgets, baseClass)
                except AttributeError: pass

                baseClass = self._widgets[baseClass][0]
            else:
                raise ValueError("baseclass resolve took too long, check custom widgets")

        except KeyError:
            raise ValueError("unknown baseclass %s" % baseClass)


    def search(self, cls):
        try:
            self._usedWidgets.add(cls)
            baseClass = self._resolveBaseclass(self._widgets[cls][0])
            DEBUG("resolved baseclass of %s: %s" % (cls, baseClass))

            return type(cls, (baseClass,),
                        {"module" : ""})

        except KeyError:
            return None

    def _writeImportCode(self):
        imports = {}
        for widget in self._usedWidgets:
            _, module = self._widgets[widget]
            imports.setdefault(module, []).append(widget)

        for module, classes in imports.items():
            write_code("from %s import %s" % (module, ", ".join(classes)))


class CompilerCreatorPolicy(object):
    def __init__(self):
        self._modules = []

    def createQtGuiWrapper(self):
        return _QtGuiWrapper

    def createQtWidgetsWrapper(self):
        return _QtWidgetsWrapper

    def createModuleWrapper(self, name, classes):
        mw = _ModuleWrapper(name, classes)
        self._modules.append(mw)
        return mw

    def createCustomWidgetLoader(self):
        cw = _CustomWidgetLoader()
        self._modules.append(cw)
        return cw

    def instantiate(self, clsObject, objectname, ctor_args, is_attribute=True, no_instantiation=False):
        return clsObject(objectname, is_attribute, ctor_args, no_instantiation)

    def invoke(self, rname, method, args):
        return method(rname, *args)

    def getSlot(self, object, slotname):
        return Literal("%s.%s" % (object, slotname))

    def _writeOutImports(self):
        for module in self._modules:
            module._writeImportCode()
