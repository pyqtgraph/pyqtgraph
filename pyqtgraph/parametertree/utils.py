from collections import OrderedDict

from pyqtgraph.parametertree.Parameter import Parameter
import pyqtgraph as pg

def get_classes(p: Parameter) -> list:
    """
    Get the classes of all the elements of a Parameter in a list.

    Parameters
    ----------
    p: Parameter
    The parameter from which the classes are going to be extracted.

    Returns
    -------
    list: A list containing a tree structure with the classes of each parameter elements.
    """
    return [p.__class__, [get_classes(c) for c in p.children()]]

def compare_parameters(p1: Parameter, p2: Parameter) -> bool:
    """
    Compare two Parameters.
    Compare the states of the parameters, then recursively compare the class of each parameter's element.

    Parameters
    ----------
    p1: Parameter
    The first parameter to compare.

    p2: Parameter
    The second parameter to compare.

    Returns
    -------
    bool: Return True if both parameters are equal, return False otherwise.

    See Also
    --------
    pg.eq, get_classes
    """
    p1_state = p1.saveState()
    p2_state = p2.saveState()

    if not pg.eq(p1_state, p2_state):
        return False

    return get_classes(p1) == get_classes(p2)
