# This Python file uses the following encoding: utf-8
import os
from typing import Dict, Tuple, Union

import pyqtgraph as pg


DEFAULT_STYLE      = 'default'
STYLE_EXTENSION    = 'pstyle'
USER_LIBRARY_PATHS = 'stylelib'

# hint of allowed style options
ConfigColorHint = Union[str,
                        int,
                        Tuple[float, float, float],
                        Tuple[float, float, float, float]]
ConfigValueHint = Union[str,
                        int,
                        float,
                        bool,
                        Tuple[float, float],
                        ConfigColorHint]
ConfigKeyHint   = str
ConfigHint      = Dict[ConfigKeyHint, ConfigValueHint]


def removeComment(line: str) -> str:
    """
    Remove comment from line
    if "#" is present, return the line up to "#" position
    if "#" not present, return empty str

    Args:
        s : line from which the comments will be removed

    Returns:
        uncommented or empty line
    """

    hash_pos = line.find('#')
    if hash_pos>0:
        return line[:hash_pos]
    # Commented line
    elif hash_pos==0:
        return ''
    # Empty line
    else:
        return ''


def getKeyVal(line: str) -> Tuple[str, str]:
    """
    Return a tuple (key, val) from a line such as:
        key : val
    """

    t = line.split(':')

    key = t[0].strip()
    val = t[1].strip()

    return key, val


def isFloat(str: str) -> bool:
    try:
        float(str)
        return True
    except ValueError:
        return False


def isTuple(str: str) -> bool:

    t = str.split(',')
    if len(t)>1:
        return True
    else:
        return False

def isBool(str: str) -> bool:

    if str in ('True', 'False'):
        return True
    else:
        return False


def validateVal(val: str) -> Union[str, float, Tuple[float, ...], bool]:

    if isFloat(val):
        return float(val)
    elif isTuple(val):
        return tuple([float(i) for i in val.split(',')])
    elif isBool(val):
        return val=='True'
    else:
        return val




def loadConfigStyle(style: str) -> dict:
    """
    Create a dictionnary from a style file.

    Args:
        style : name of the style to load

    Returns:
        dict: A mapping of the key : val written in the style file
    """

    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), USER_LIBRARY_PATHS, '{}.{}'.format(style, STYLE_EXTENSION))
    styleDict = {}
    with open(file, encoding='utf-8', mode='r') as f:

        for line in f:
            line = removeComment(line)
            if line=='':
                continue

            key, val = getKeyVal(line)

            styleDict[key] = validateVal(val)

    return styleDict


def loadDefaultStyle() -> dict:

    return loadConfigStyle(DEFAULT_STYLE)


def use(styleName: str) -> None:

    styleDict = loadConfigStyle(styleName)

    pg.configStyle.update(styleDict)

# Currently hint not correct, because of circular import...
def initItemStyle(item,
                  itemName: str,
                  configStyle: ConfigHint) -> None:
    """
    Add to internal Item opts attribute all Item style options from the
    stylesheet.
    """

    for key, val in configStyle.items():
        if key[:len(itemName)]==itemName:
            fun = getattr(item, 'set{}{}'.format(key[len(itemName)+1:][:1].upper(),
                                                    key[len(itemName)+1:][1:]))
            fun(val)
