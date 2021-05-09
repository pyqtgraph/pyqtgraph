import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtCore, QtGui

import pytest

def test_registration():
    reg = pg.functions.colorRegistry()
    color = reg.getRegisteredColor('p0')
    color_copy = QtGui.QColor(color) # make a copy
    # confirm that a registration was assigned:
    assert hasattr(color,'registration')
    # confirm that the registration is not copied:
    assert not hasattr(color_copy,'registration')

def test_specification_formats():
    next_reg = None
    reg = pg.functions.colorRegistry()
    reg.registered_objects = {} # force-clear registry
    obj_list = [] # start building a list of test objects
    assert len(reg.registered_objects) == len(obj_list) # registry should be empty
    args = ( 
        ( True, ('p0', 1, 255) ),
        ( True, ('p1', None, 192) ),
        ( True, ('p2', 2) ),
        ( False, ('#00112233') ),
        ( False, '#012' ),
        ( False, '#xxx' ), # QColor converts any invalid string to #000000 
        ( False, 'µ' ), # QColor converts any invalid string to #000000 
        ( None, 1e129 ),
        ( None, ( 1.0, 0.5, 0.2 ) ),
        ( None, ( 5, 12 ) ),
        ( None, {'1':2} )
    )
    for expectation, desc in args: # test pen registration
        # print( expectation, desc )
        pen = reg.getRegisteredPen(desc)
        if expectation is None:
            # print( desc, pen )
            assert pen is None # does not match input patterns
        elif expectation: # mathces input pattern for registered pen
            if next_reg is None: # initialize, then watch it count up
                next_reg = pen.registration
            assert pen.registration == next_reg
            next_reg += 1
            obj_list.append(pen)
        else: # matches (hex) input pattern for regular QColor
            assert not hasattr(pen,'registration')
        del pen

    args = ( 
        ( True, ('p0', 255) ),
        ( True, 'p1'),
        ( True, ('p2', 0.1) ),
        ( False, ('#00112233') ),
        ( False, '#001122' ),
        ( None, 1e129 ),
        ( None, ( 1.0, 0.5, 0.2 ) ),
        ( None, ( 5, 12 ) )
    )
    for expectation, desc in args: # test brush registration
        brush = reg.getRegisteredBrush(desc)
        if expectation is None:
            assert brush is None
        elif expectation:
            assert brush.registration == next_reg
            next_reg += 1
            obj_list.append(brush)
        else:
            assert not hasattr(brush,'registration')
        del brush

    for expectation, desc in args: # test color registration
        color = reg.getRegisteredColor(desc)
        if expectation is None:
            assert color is None
        if expectation:
            assert color.registration == next_reg
            next_reg += 1
            obj_list.append(color)
        else:
            assert not hasattr(color,'registration')
        del color
    assert len(reg.registered_objects) == len(obj_list)

    new_colors = (
        ('p0', QtGui.QColor('#000001') ),
        ('p1', QtGui.QColor('#000002') ),
        ('p2', QtGui.QColor('#000003') )
    )
    updated_colors = reg.colors().copy()
    for key, qcol in new_colors:
        updated_colors[key] = qcol
    reg.redefinePalette(updated_colors)

    for idx, obj in enumerate(obj_list):
        col_idx = idx % 3 # colors 'p0' to 'p2' repeat for pen, brush and color
        qcol = obj.color() if hasattr(obj,'color') else obj
        # print(qcol.name(),' -- ', new_colors[col_idx][1].name() ) 
        assert qcol.rgb() == new_colors[col_idx][1].rgb()
    del obj
    del qcol # do not keep extra references

    assert len(reg.registered_objects) == len(obj_list)
    obj_list = [] # delete all objects
    print('remaining registered objects:', reg.registered_objects )
    assert (
        len(reg.registered_objects) == 0 or # All cleared by finalize calls, except that 
        len(reg.registered_objects) == 1    # PySide seems to be left with one surviving reference
    )

def test_mkColor_edge_case_specifications():
    color = fn.mkColor(1e129)
    assert color.name() == '#ffffff' # large float should be clipped and treated as 0-1 grayscale value
    assert not hasattr(color,'registration') # large float input should not yield a registered color

    try:
        color = fn.mkColor(True)
        assert False # mkColor(True) is expected to raise an error
    except TypeError:
        pass

    try:
        color = fn.mkColor(None)
        assert False # mkColor(True) is expected to raise an error
    except TypeError:
        pass
