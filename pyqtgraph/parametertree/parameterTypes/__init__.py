from .action import ActionParameter
from .basetypes import WidgetParameterItem, SimpleParameter, GroupParameter
from .calendar import CalendarParameter
from .file import FileParameter
from .font import FontParameter
from .list import ListParameter
from .pen import PenParameter
from .progress import ProgressBarParameter
from .qtenum import QtEnumParameter
from .slider import SliderParameter
from .text import TextParameter
from ..Parameter import registerParameterType

registerParameterType('int', SimpleParameter, override=True)
registerParameterType('float', SimpleParameter, override=True)
registerParameterType('bool', SimpleParameter, override=True)
registerParameterType('str', SimpleParameter, override=True)
registerParameterType('color', SimpleParameter, override=True)
registerParameterType('colormap', SimpleParameter, override=True)

registerParameterType('group', GroupParameter, override=True)

registerParameterType('list', ListParameter, override=True)
registerParameterType('action', ActionParameter, override=True)
registerParameterType('text', TextParameter, override=True)
registerParameterType('pen', PenParameter, override=True)
registerParameterType('progress', ProgressBarParameter, override=True)
registerParameterType('file', FileParameter, override=True)
registerParameterType('slider', SliderParameter, override=True)
registerParameterType('calendar', CalendarParameter, override=True)
registerParameterType('font', FontParameter, override=True)
# qtenum is a bit specific, hold off on registering for now
