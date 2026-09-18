from enum import StrEnum

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