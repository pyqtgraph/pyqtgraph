from pyqtgraph import configfile
import numpy as np
import tempfile, os

def test_longArrays():
    """
    Test config saving and loading of long arrays.
    """
    tmp = tempfile.mktemp(".cfg")

    arr = np.arange(20)
    configfile.writeConfigFile({'arr':arr}, tmp)
    config = configfile.readConfigFile(tmp)
    
    assert all(config['arr'] == arr)

    os.remove(tmp)

def test_multipleParameters():
    """
    Test config saving and loading of multiple parameters.
    """
    tmp = tempfile.mktemp(".cfg")

    par1 = [1,2,3]
    par2 = "Test"
    par3 = {'a':3,'b':'c'}

    configfile.writeConfigFile({'par1':par1, 'par2':par2, 'par3':par3}, tmp)
    config = configfile.readConfigFile(tmp)
    
    assert config['par1'] == par1
    assert config['par2'] == par2
    assert config['par3'] == par3

    os.remove(tmp)
