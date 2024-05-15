from ..Parameter import registerParameterType
from .action import ActionParameter, ActionParameterItem
from .actiongroup import ActionGroup, ActionGroupParameter, ActionGroupParameterItem
from .basetypes import (
    GroupParameter,
    GroupParameterItem,
    SimpleParameter,
    WidgetParameterItem,
)
from .bool import BoolParameterItem
from .calendar import CalendarParameter, CalendarParameterItem
from .checklist import ChecklistParameter, ChecklistParameterItem
from .color import ColorParameter, ColorParameterItem
from .colormap import ColorMapParameter, ColorMapParameterItem
from .colormaplut import ColorMapLutParameter, ColorMapLutParameterItem
from .file import FileParameter, FileParameterItem
from .font import FontParameter, FontParameterItem
from .list import ListParameter, ListParameterItem
from .numeric import NumericParameterItem
from .pen import PenParameter, PenParameterItem
from .progress import ProgressBarParameter, ProgressBarParameterItem
from .qtenum import QtEnumParameter
from .slider import SliderParameter, SliderParameterItem
from .str import StrParameterItem
from .text import TextParameter, TextParameterItem

registerParameterType('group',        GroupParameter,       override=True)
# Keep actiongroup private for now, mainly useful for Interactor but not externally
registerParameterType('_actiongroup', ActionGroupParameter, override=True)

registerParameterType('action',    ActionParameter,      override=True)
registerParameterType('bool',      SimpleParameter,      override=True)
registerParameterType('calendar',  CalendarParameter,    override=True)
registerParameterType('checklist', ChecklistParameter,   override=True)
registerParameterType('cmaplut',   ColorMapLutParameter, override=True)
registerParameterType('color',     ColorParameter,       override=True)
registerParameterType('colormap',  ColorMapParameter,    override=True)
registerParameterType('file',      FileParameter,        override=True)
registerParameterType('float',     SimpleParameter,      override=True)
registerParameterType('font',      FontParameter,        override=True)
registerParameterType('int',       SimpleParameter,      override=True)
registerParameterType('list',      ListParameter,        override=True)
registerParameterType('pen',       PenParameter,         override=True)
registerParameterType('progress',  ProgressBarParameter, override=True)
# qtenum is a bit specific, hold off on registering for now
registerParameterType('slider',    SliderParameter,      override=True)
registerParameterType('str',       SimpleParameter,      override=True)
registerParameterType('text',      TextParameter,        override=True)
