from ..Parameter import registerParameterItemType, registerParameterType
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

registerParameterItemType('bool',  BoolParameterItem,    SimpleParameter, override=True)
registerParameterItemType('float', NumericParameterItem, SimpleParameter, override=True)
registerParameterItemType('int',   NumericParameterItem, SimpleParameter, override=True)
registerParameterItemType('str',   StrParameterItem,     SimpleParameter, override=True)

registerParameterType('group',        GroupParameter,       override=True)
# Keep actiongroup private for now, mainly useful for Interactor but not externally
registerParameterType('_actiongroup', ActionGroupParameter, override=True)

registerParameterType('action',    ActionParameter,      override=True)
registerParameterType('calendar',  CalendarParameter,    override=True)
registerParameterType('checklist', ChecklistParameter,   override=True)
registerParameterType('color',     ColorParameter,       override=True)
registerParameterType('colormap',  ColorMapParameter,    override=True)
registerParameterType('file',      FileParameter,        override=True)
registerParameterType('font',      FontParameter,        override=True)
registerParameterType('list',      ListParameter,        override=True)
registerParameterType('pen',       PenParameter,         override=True)
registerParameterType('progress',  ProgressBarParameter, override=True)
# qtenum is a bit specific, hold off on registering for now
registerParameterType('slider',    SliderParameter,      override=True)
registerParameterType('text',      TextParameter,        override=True)
