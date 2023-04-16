# This Python file uses the following encoding: utf-8
import ast
from configparser import ConfigParser, Interpolation
import os
from typing import Dict, List, Tuple, Union

import pyqtgraph as pg


DEFAULT_STYLE      = 'default'
# DEFAULT_STYLE      = 'matplotlib'
STYLE_EXTENSION    = 'pstyle'
USER_LIBRARY_PATHS = 'stylelib'

# hint of allowed style options
ConfigColorHint = Union[str, # 'c' one of: r, g, b, c, m, y, k, w or or "#RGB" or "#RGBA" or "#RRGGBB" or "#RRGGBBAA"
                        int, # see :func:`intColor() <pyqtgraph.intColor>`
                        float, # greyscale, 0.0-1.0
                        Tuple[int, int, int, int], # list of integers 0-255  (R, G, B, A)
                        Tuple[int, int, int]] # list of integers 0-255  (R, G, B)
ConfigLinestyleHint = Union[str, int]
ConfigValueHint = Union[str, # example: font weight
                        int,  # example: tick alpha
                        float,  # example: label font size
                        bool, # example: autoExpandTextSpace
                        List[int], # example: tickTextOffset
                        List[bool], # example: stopAxisAtTick
                        List[Tuple[float, float]], # example: textFillLimits
                        ConfigColorHint]
ConfigKeyHint   = str
ConfigHint      = Dict[ConfigKeyHint, Dict[ConfigKeyHint, ConfigValueHint]]

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


class TypedInterpolation(Interpolation):
    def before_get(self, parser, section, option, value, defaults):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return super().before_get(parser, section, option, value, defaults)
typedConfig = ConfigParser(interpolation=TypedInterpolation())
# To preserve the case of the style options
typedConfig.optionxform = str # type: ignore


def loadConfigStyle(style: str) -> dict:
    """
    Create a dictionnary from a style file.

    Args:
        style : name of the style to load

    Returns:
        dict: A mapping of the key : val written in the style file
    """

    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), USER_LIBRARY_PATHS, '{}.{}'.format(style, STYLE_EXTENSION))
    with open(file, encoding='utf-8', mode='r') as f:
        typedConfig.read_string(f.read())

    return {section_name: dict(typedConfig[section_name]) for section_name in typedConfig.sections()}


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

    Parameters
    ----------
    item
        The item to init.
    itemName
        Name of the item in the style file.
    configStyle
        The current config Style dictionnary
    """

    for key, val in configStyle[itemName].items():
        fun = getattr(item, 'set{}{}'.format(key[:1].upper(),
                                             key[1:]))
        fun(val)
