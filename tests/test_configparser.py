from pyqtgraph import configfile
import numpy as np

def test_longArrays(tmpdir):
    """
    Test config saving and loading of long arrays.
    """
    arr = np.arange(20)

    tf = tmpdir.join("config.cfg")
    configfile.writeConfigFile({'arr': arr}, tf)
    config = configfile.readConfigFile(tf)
    assert all(config['arr'] == arr)

def test_multipleParameters(tmpdir):
    """
    Test config saving and loading of multiple parameters.
    """

    par1 = [1,2,3]
    par2 = "Test"
    par3 = {'a':3,'b':'c'}

    tf = tmpdir.join("config.cfg")
    configfile.writeConfigFile({'par1':par1, 'par2':par2, 'par3':par3}, tf)
    config = configfile.readConfigFile(tf)

    assert config['par1'] == par1
    assert config['par2'] == par2
    assert config['par3'] == par3
