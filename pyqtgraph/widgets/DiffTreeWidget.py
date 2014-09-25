# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore
from ..pgcollections import OrderedDict
from .DataTreeWidget import DataTreeWidget
from .. import functions as fn
import types, traceback
import numpy as np

__all__ = ['DiffTreeWidget']


class DiffTreeWidget(QtGui.QWidget):
    """
    Widget for displaying differences between hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """
    def __init__(self, parent=None, a=None, b=None):
        QtGui.QWidget.__init__(self, parent)
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)
        self.trees = [DataTreeWidget(self), DataTreeWidget(self)]
        for t in self.trees:
            self.layout.addWidget(t)
        if a is not None:
            self.setData(a, b)
    
    def setData(self, a, b):
        """
        Set the data to be compared in this widget.
        """
        self.data = (a, b)
        self.trees[0].setData(a)
        self.trees[1].setData(b)
        
        return self.compare(a, b)
        
    def compare(self, a, b, path=()):
        """
        Compare data structure *a* to structure *b*. 
        
        Return True if the objects match completely. 
        Otherwise, return a structure that describes the differences:
        
            { 'type': bool
              'len': bool,
              'str': bool,
              'shape': bool,
              'dtype': bool,
              'mask': array,
              }
        
                
        """
        bad = (255, 200, 200)
        diff = []
        # generate typestr, desc, childs for each object
        typeA, descA, childsA, _ = self.trees[0].parse(a)
        typeB, descB, childsB, _ = self.trees[1].parse(b)
        
        if typeA != typeB:
            self.setColor(path, 1, bad)
        if descA != descB:
            self.setColor(path, 2, bad)
            
        if isinstance(a, dict) and isinstance(b, dict):
            keysA = set(a.keys())
            keysB = set(b.keys())
            for key in keysA - keysB:
                self.setColor(path+(key,), 0, bad, tree=0)
            for key in keysB - keysA:
                self.setColor(path+(key,), 0, bad, tree=1)
            for key in keysA & keysB:
                self.compare(a[key], b[key], path+(key,))
            
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            for i in range(max(len(a), len(b))):
                if len(a) <= i:
                    self.setColor(path+(i,), 0, bad, tree=1)
                elif len(b) <= i:
                    self.setColor(path+(i,), 0, bad, tree=0)
                else:
                    self.compare(a[i], b[i], path+(i,))
                    
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
            tableNodes = [tree.nodes[path].child(0) for tree in self.trees]
            if a.dtype.fields is None and b.dtype.fields is None:
                eq = self.compareArrays(a, b)
                if not np.all(eq):
                    for n in tableNodes:
                        n.setBackground(0, fn.mkBrush(bad))
                #for i in np.argwhere(~eq):
                    
            else:
                for i,k in enumerate(info.dtype.fields.keys()):
                    eq = self.compareArrays(a[k], b[k])
                    if not np.all(eq):
                        for n in tableNodes:
                            n.setBackground(0, fn.mkBrush(bad))
                    #for j in np.argwhere(~eq):
                    
        # dict: compare keys, then values where keys match
        # list: 
        # array: compare elementwise for same shape

    def compareArrays(self, a, b):
        intnan = -9223372036854775808  # happens when np.nan is cast to int
        anans = np.isnan(a) | (a == intnan)
        bnans = np.isnan(b) | (b == intnan)
        eq = anans == bnans
        mask = ~anans
        eq[mask] = np.allclose(a[mask], b[mask])
        return eq
    
    def setColor(self, path, column, color, tree=None):
        brush = fn.mkBrush(color)
        
        # Color only one tree if specified.
        if tree is None:
            trees = self.trees
        else:
            trees = [self.trees[tree]]
        
        for tree in trees:
            item = tree.nodes[path]
            item.setBackground(column, brush)
    
    def _compare(self, a, b):
        """
        Compare data structure *a* to structure *b*. 
        """
        # Check test structures are the same
        assert type(info) is type(expect)
        if hasattr(info, '__len__'):
            assert len(info) == len(expect)
            
        if isinstance(info, dict):
            for k in info:
                assert k in expect
            for k in expect:
                assert k in info
                self.compare_results(info[k], expect[k])
        elif isinstance(info, list):
            for i in range(len(info)):
                self.compare_results(info[i], expect[i])
        elif isinstance(info, np.ndarray):
            assert info.shape == expect.shape
            assert info.dtype == expect.dtype
            if info.dtype.fields is None:
                intnan = -9223372036854775808  # happens when np.nan is cast to int
                inans = np.isnan(info) | (info == intnan)
                enans = np.isnan(expect) | (expect == intnan)
                assert np.all(inans == enans)
                mask = ~inans
                assert np.allclose(info[mask], expect[mask])
            else:
                for k in info.dtype.fields.keys():
                    self.compare_results(info[k], expect[k])
        else:
            try:
                assert info == expect
            except Exception:
                raise NotImplementedError("Cannot compare objects of type %s" % type(info))
    