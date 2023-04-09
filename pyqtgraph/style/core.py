# This Python file uses the following encoding: utf-8
import os
from typing import Dict, Tuple, Union

import pyqtgraph as pg


# DEFAULT_STYLE      = 'default'
DEFAULT_STYLE      = 'matplotlib'
STYLE_EXTENSION    = 'pstyle'
USER_LIBRARY_PATHS = 'stylelib'

# hint of allowed style options
ConfigColorHint = Union[str,
                        int,
                        Tuple[float, float, float],
                        Tuple[float, float, float, float]]
ConfigLinestyleHint = Union[str, int]
ConfigValueHint = Union[str,
                        int,
                        float,
                        bool,
                        Tuple[float, float],
                        ConfigColorHint]
ConfigKeyHint   = str
ConfigHint      = Dict[ConfigKeyHint, ConfigValueHint]

parseLineStyleDict = {"-" :   1,
                      "--" :  2,
                      ":" :   3,
                      "-." :  4,
                      "-.." : 5}
def parseLineStyle(linestyle: ConfigLinestyleHint) -> int:
    """
    Parse the given linestyle, string or integerm into an accepted Qt,
    linestyle, integer only.

    Correspondence between linestyle:
        * "-", "1"   -> A plain line.
        * "--, "2"   -> Dashes separated by a few pixels.
        * ":", "3"   -> Dots separated by a few pixels.
        * "-.", "4"  -> Alternate dots and dashes.
        * "-..", "5" -> One dash, two dots, one dash, two dots.
    """

    if isinstance(linestyle, str):
        if linestyle in parseLineStyleDict.keys():
            return parseLineStyleDict[linestyle]
        else:
            raise ValueError('Given "linestyle" argument:{} must be "-", "--", ":", "-." or, "-..".'.format(linestyle))
    elif isinstance(linestyle, int):
        if linestyle in parseLineStyleDict.values():
            return linestyle
        else:
            raise ValueError('Given "linestyle" argument:{} must be 0, 1, 2, 3 or, 4.'.format(linestyle))
    else:
        raise ValueError('Given "linestyle" argument:{} must be a string or a int'.format(linestyle))


def removeComment(line: str) -> str:
    """
    Remove comment from line
    if "#" is present:
        if "#" position is 0, we assume commented line
            return empty line
        if "#" position is 1 after a ":", we assume a color in hex
            return line up to the second "#".
        else we assume comments about the style itself
            return the line up to "#" position
    if "#" not present, return empty str

    Args:
        s : line from which the comments will be removed

    Returns:
        uncommented or empty line
    """

    hash_pos = line.find('#')
    # Commented line
    if hash_pos==0:
        return ''
    elif hash_pos>0:
        hash_pos_ = line[::-1].find('#')
        return line[:-hash_pos_-1]
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
