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

from ..exceptions import WidgetPluginError


def load_plugin(plugin, plugin_globals, plugin_locals):
    """ Load the given plugin (which is an open file).  Return True if the
    plugin was loaded, or False if it wanted to be ignored.  Raise an exception
    if there was an error.
    """

    try:
        exec(plugin.read(), plugin_globals, plugin_locals)
    except ImportError:
        return False
    except Exception as e:
        raise WidgetPluginError("%s: %s" % (e.__class__, str(e)))

    return True
