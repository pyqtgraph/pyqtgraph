import numpy as np

from pyqtgraph import configfile


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


def test_duplicate_keys_error(tmpdir):
    """
    Test that an error is raised when duplicate keys are present in the config file.
    """

    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write('a: 1\n')
        f.write('a: 2\n')

    try:
        configfile.readConfigFile(tf)
    except configfile.ParseError as e:
        assert 'Duplicate key' in str(e)
    else:
        assert False, "Expected ParseError"


def test_line_numbers_acconut_for_comments_and_blanks(tmpdir):
    """
    Test that line numbers in ParseError account for comments and blank lines.
    """

    tf = tmpdir.join("config.cfg")
    with open(tf, 'w') as f:
        f.write('a: 1\n')
        f.write('\n')
        f.write('# comment\n')
        f.write('a: 2\n')

    try:
        configfile.readConfigFile(tf)
    except configfile.ParseError as e:
        assert 'at line 4' in str(e)
    else:
        assert False, "Expected ParseError"
