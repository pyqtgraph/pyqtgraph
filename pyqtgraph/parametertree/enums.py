try:
    from enum import StrEnum
except ImportError:
    # Fallback for Python < 3.11 to maintain strict compatibility without dependencies
    from enum import Enum
    class StrEnum(str, Enum):
        pass

class ParameterChangeType(StrEnum):
    """Enumeration of native change types emitted by PyQtGraph Parameters."""
    VALUE = "value"
    LIMITS = "limits"
    OPTIONS = "options"
    CHILD_ADDED = "childAdded"
    PARENT = "parent"
    NAME = "name"
    READ_ONLY = "readOnly"
    VISIBLE = "visible"
    CONTEXT_MENU = "contextMenu"