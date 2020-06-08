# -*- coding: utf-8 -*-
import pytest
import pyqtgraph as pg
from pyqtgraph.exporters import HDF5Exporter
import numpy as np
from numpy.testing import assert_equal
import h5py
import os


@pytest.fixture
def tmp_h5(tmp_path):
    yield tmp_path / "data.h5"


@pytest.mark.parametrize("combine", [False, True])
def test_HDF5Exporter(tmp_h5, combine):
    # Basic test of functionality: multiple curves with shared x array. Tests
    # both options for stacking the data (columnMode).
    x = np.linspace(0, 1, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt = pg.plot()
    plt.plot(x=x, y=y1)
    plt.plot(x=x, y=y2)

    ex = HDF5Exporter(plt.plotItem)

    if combine:
        ex.parameters()['columnMode'] = '(x,y,y,y) for all plots'

    ex.export(fileName=tmp_h5)

    with h5py.File(tmp_h5, 'r') as f:
        # should be a single dataset with the name of the exporter
        dset = f[ex.parameters()['Name']]
        assert isinstance(dset, h5py.Dataset)

        if combine:
            assert_equal(np.array([x, y1, y2]), dset)
        else:
            assert_equal(np.array([x, y1, x, y2]), dset)


def test_HDF5Exporter_unequal_lengths(tmp_h5):
    # Test export with multiple curves of different size. The exporter should
    # detect this and create multiple hdf5 datasets under a group.
    x1 = np.linspace(0, 1, 10)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 1, 100)
    y2 = np.cos(x2)

    plt = pg.plot()
    plt.plot(x=x1, y=y1, name='plot0')
    plt.plot(x=x2, y=y2)

    ex = HDF5Exporter(plt.plotItem)
    ex.export(fileName=tmp_h5)

    with h5py.File(tmp_h5, 'r') as f:
        # should be a group with the name of the exporter
        group = f[ex.parameters()['Name']]
        assert isinstance(group, h5py.Group)

        # should be a dataset under the group with the name of the PlotItem
        assert_equal(np.array([x1, y1]), group['plot0'])

        # should be a dataset under the group with a default name that's the
        # index of the curve in the PlotItem
        assert_equal(np.array([x2, y2]), group['1'])
