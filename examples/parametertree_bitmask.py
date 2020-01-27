# -*- coding: utf-8 -*-
"""
This example demonstrates the use of pyqtgraph's parametertree system
with the 'bitmask' type (BitmaskParameter) for handlig enum.IntFlag,
available from Python 3.6 https://docs.python.org/3/library/enum.html#enum.IntFlag

"""

import initExample ## Add path to library (just for examples; you do not need this)

#import pyqtgraph as pg
import sys
from pyqtgraph.Qt import QtWidgets, QtGui
from pyqtgraph.parametertree import BitmaskParameter # this import registers 'bitmask' as a Parameter type
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree import ParameterTree

from enum import IntFlag
class ChannelMask(IntFlag):
    '''Represents a choice of zero, one or multiple channels.'''
    A = 0b0001 # only the first channel
    B = 0b0010 # only the second channel
    C = 0b0100 # only the third channel
    D = 0b1000 # only the fourth channel

if __name__ == '__main__':
    
    # Different forms of explicit construction of a BitmaskParameter
#    structure = [{'type': 'group', 'name': 'Examples', 'children': [
#            {'type': 'float', 'name': 'A number', 'value': 1.23},
#            BitmaskParameter(name='Example 1', value=ChannelMask.A|ChannelMask.B),
#            BitmaskParameter(name='Example 2', value=ChannelMask.A|ChannelMask.B, numeric=False), # skip the numeric indicator
#            BitmaskParameter(name='Example 3', value=3, values=dict(ChannelMask.__members__)), # allows value to be a plain integer
#            BitmaskParameter(name='Empty start', values=dict(ChannelMask.__members__)), # will have 0 as default value
#            BitmaskParameter(name='Restricted', values={f.name: f.value for f in list(ChannelMask)[0:2]}), # show only the first two bits
#            {'type': 'str', 'name': 'Something else', 'value': ''}
#        ]}]
    
    # Indirect construction from dict
    structure = [{'type': 'group', 'name': 'Examples', 'children': [
            {'type': 'float', 'name': 'A number', 'value': 1.23},
            {'type': 'bitmask', 'name': 'Channels', 'value': ChannelMask.A|ChannelMask.B},
            {'type': 'bitmask', 'name': 'Channels without showing integer', 'numeric': False, # don't show the numeric indicator
                 'value': 3, 'values': dict(ChannelMask.__members__)}, # accepts plain integer value (since 'values' is given)
            {'type': 'bitmask', 'name': 'Empty start', 'values': dict(ChannelMask.__members__)},
            {'type': 'bitmask', 'name': 'Restricted', 'values': {f.name: f.value for f in list(ChannelMask)[0:2]}}, # show only the first two bits
            {'type': 'str', 'name': 'Something else', 'value': ''}
        ]}]
    ## Create tree of Parameter objects
    paramTree = Parameter.create(name='root', type='group', children=structure)
    
    
    app = QtWidgets.QApplication([])
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True) 
    win = QtGui.QWidget()
    t = ParameterTree(win)
    t.setParameters(paramTree, showTop=False)
    t.resize(350, 550)
    win.setGeometry(200, 200, 400, 600)
    win.show()
    sys.exit(app.exec_())