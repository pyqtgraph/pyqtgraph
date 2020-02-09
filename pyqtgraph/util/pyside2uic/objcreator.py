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
import os.path

from .exceptions import NoSuchWidgetError, WidgetPluginError

if sys.hexversion >= 0x03000000:
    from .port_v3.load_plugin import load_plugin
else:
    from .port_v2.load_plugin import load_plugin


# The list of directories that are searched for widget plugins.  This is
# exposed as part of the API.
widgetPluginPath = [os.path.join(os.path.dirname(__file__), 'widget-plugins')]


MATCH = True
NO_MATCH = False
MODULE = 0
CW_FILTER = 1


class QObjectCreator(object):
    def __init__(self, creatorPolicy):
        self._cpolicy = creatorPolicy

        self._cwFilters = []
        self._modules = [self._cpolicy.createQtWidgetsWrapper(),
                         self._cpolicy.createQtGuiWrapper()]

        # Get the optional plugins.
        for plugindir in widgetPluginPath:
            try:
                plugins = os.listdir(plugindir)
            except:
                plugins = []

            for filename in plugins:
                if not filename.endswith('.py') or filename == '__init__.py':
                    continue

                filename = os.path.join(plugindir, filename)

                plugin_globals = {
                    "MODULE": MODULE,
                    "CW_FILTER": CW_FILTER,
                    "MATCH": MATCH,
                    "NO_MATCH": NO_MATCH}

                plugin_locals = {}

                if load_plugin(open(filename), plugin_globals, plugin_locals):
                    pluginType = plugin_locals["pluginType"]
                    if pluginType == MODULE:
                        modinfo = plugin_locals["moduleInformation"]()
                        self._modules.append(self._cpolicy.createModuleWrapper(*modinfo))
                    elif pluginType == CW_FILTER:
                        self._cwFilters.append(plugin_locals["getFilter"]())
                    else:
                        raise WidgetPluginError("Unknown plugin type of %s" % filename)

        self._customWidgets = self._cpolicy.createCustomWidgetLoader()
        self._modules.append(self._customWidgets)

    def createQObject(self, classname, *args, **kwargs):
        classType = self.findQObjectType(classname)
        if classType:
            return self._cpolicy.instantiate(classType, *args, **kwargs)
        raise NoSuchWidgetError(classname)

    def invoke(self, rname, method, args=()):
        return self._cpolicy.invoke(rname, method, args)

    def findQObjectType(self, classname):
        for module in self._modules:
            w = module.search(classname)
            if w is not None:
                return w
        return None

    def getSlot(self, obj, slotname):
        return self._cpolicy.getSlot(obj, slotname)

    def addCustomWidget(self, widgetClass, baseClass, module):
        for cwFilter in self._cwFilters:
            match, result = cwFilter(widgetClass, baseClass, module)
            if match:
                widgetClass, baseClass, module = result
                break

        self._customWidgets.addCustomWidget(widgetClass, baseClass, module)
