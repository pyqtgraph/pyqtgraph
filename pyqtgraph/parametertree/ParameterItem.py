import warnings

from .. import icons
from ..Qt import QtCore, QtGui, QtWidgets

translate = QtCore.QCoreApplication.translate


class _MenuActionHandler(QtCore.QObject):
    """QObject helper that receives triggered signals from QActions in a menu.

    The handler avoids lambda closures in signal connections by storing
    the path directly on the action (``action.pathForTriggered``) and using
    ``sender()`` to retrieve it at call time.  This prevents unintended strong
    reference cycles between Qt objects.

    Parameters
    ----------
    callback : callable
        Called with a single argument — the ``pathForTriggered`` tuple of the
        triggered action.
    """

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def onTriggered(self):
        action = self.sender()
        if action is not None and hasattr(action, 'pathForTriggered'):
            self._callback(action.pathForTriggered)


def build_menu_from_iterable(menu, items, handler, path=()):
    """Recursively populate *menu* from *items*.

    Each leaf action is connected to ``handler.onTriggered`` and has its
    ``pathForTriggered`` attribute set to the full tuple path from the root.

    Parameters
    ----------
    menu : QMenu
        The menu (or submenu) to populate.
    items : dict | list | tuple
        Structure describing the menu.  Each element is either:

        - a plain ``str`` — a leaf action whose display text and path
          component are both the string.
        - a ``dict`` — key/value pairs interpreted as:

          ============  ====================================================
          value type    meaning
          ============  ====================================================
          falsy         leaf; display text = key, path component = key
          ``str``       leaf; display text = value, path component = key
                        (human-readable alias, e.g. ``{"internalName": "Label"}``)
          non-empty     submenu named *key*; recurse into *value*
          dict/list/
          tuple
          ============  ====================================================

    handler : _MenuActionHandler
        QObject whose ``onTriggered`` slot is connected to every leaf action.
    path : tuple
        Path prefix accumulated during recursion; callers should omit this.
    """
    if isinstance(items, dict):
        for key, value in items.items():
            _menu_handle_item(menu, key, value, handler, path)
    elif isinstance(items, (list, tuple)):
        for item in items:
            if isinstance(item, dict):
                for key, value in item.items():
                    _menu_handle_item(menu, key, value, handler, path)
            elif isinstance(item, str):
                _menu_add_leaf(menu, item, handler, path + (item,))


def _menu_handle_item(menu, key, value, handler, path):
    new_path = path + (key,)
    if isinstance(value, (dict, list, tuple)) and value:
        submenu = menu.addMenu(key)
        build_menu_from_iterable(submenu, value, handler, new_path)
    elif isinstance(value, str):
        _menu_add_leaf(menu, value, handler, new_path)
    else:
        _menu_add_leaf(menu, key, handler, new_path)


def _menu_add_leaf(menu, display, handler, path):
    action = menu.addAction(display)
    action.pathForTriggered = path
    action.triggered.connect(handler.onTriggered)


#: Default set of built-in actions shown in the ctrl button menu.
#: Pass a subset as the ``ctrlActions`` parameter option to restrict the menu.
#: Valid values: ``'default'``, ``'setDefault'``, ``'enabled'``, ``'readonly'``,
#: ``'rename'``, ``'remove'``.  For ``'rename'`` and ``'remove'``, including the
#: key in ``ctrlActions`` is sufficient — ``renamable`` / ``removable`` opts are
#: not required, though either alone is also enough.
DEFAULT_CTRL_ACTIONS = frozenset({'default', 'setDefault', 'enabled', 'readonly'})


class _CtrlMenu(QtWidgets.QMenu):
    """QMenu that stays open when an action marked with ``persistentMenu`` is triggered."""

    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action is not None and action.property("persistentMenu"):
            action.trigger()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CtrlButton(QtWidgets.QToolButton):
    """Self-contained ctrl button for :class:`ParameterItem`.

    Owns the :class:`_CtrlMenu` and delegates menu population to
    :meth:`ParameterItem.populateCtrlMenu` on the associated item.
    The item is responsible for filling the menu; this class only manages
    widget appearance and menu lifecycle.
    """

    def __init__(self, param_item):
        super().__init__()
        self._item = param_item
        self.setFixedWidth(20)
        self.setFixedHeight(20)
        self.setIcon(icons.getGraphIcon('ctrl'))
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        # hide the built-in drop-arrow so the icon fills the button cleanly
        self.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        self._menu = _CtrlMenu()  # keep a Python reference to prevent GC
        self._menu.aboutToShow.connect(param_item.populateCtrlMenu)
        self.setMenu(self._menu)


class ParameterItem(QtWidgets.QTreeWidgetItem):
    """
    Abstract ParameterTree item.
    Used to represent the state of a Parameter from within a ParameterTree.
    
      - Sets first column of item to name
      - generates context menu if item is renamable or removable
      - handles child added / removed events
      - provides virtual functions for handling changes from parameter

    Subclasses that display a value widget may call :meth:`makeCtrlButton` to
    add a ctrl button (gear icon) with a menu that exposes built-in actions
    (Reset to default, Set as default, Enable/Disable, Lock/Unlock, Rename,
    Remove).  Override :meth:`populateCtrlMenu` to customise the menu.

    For more ParameterItem types, see ParameterTree.parameterTypes module.
    """

    def __init__(self, param, depth=0):
        QtWidgets.QTreeWidgetItem.__init__(self, [param.title(), ''])

        self.param = param
        self.param.registerItem(self)  ## let parameter know this item is connected to it (for debugging)
        self.depth = depth
        
        param.sigValueChanged.connect(self.valueChanged)
        param.sigChildAdded.connect(self.childAdded)
        param.sigChildRemoved.connect(self.childRemoved)
        param.sigNameChanged.connect(self.nameChanged)
        param.sigLimitsChanged.connect(self.limitsChanged)
        param.sigDefaultChanged.connect(self.defaultChanged)
        param.sigOptionsChanged.connect(self.optsChanged)
        param.sigParentChanged.connect(self.parentChanged)
        
        self.updateFlags()

        ## flag used internally during name editing
        self.ignoreNameColumnChange = False

    def updateFlags(self):
        ## called when Parameter opts changed
        opts = self.param.opts
        ctrl = opts.get('ctrlActions', DEFAULT_CTRL_ACTIONS)
        renamable = opts.get('renamable', False) or 'rename' in ctrl

        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        if renamable:
            if opts.get('title', None) is not None:
                raise Exception("Cannot make parameter with both title != None and renamable == True.")
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        
        ## handle movable / dropEnabled options
        if opts.get('movable', False):
            flags |= QtCore.Qt.ItemFlag.ItemIsDragEnabled
        if opts.get('dropEnabled', False):
            flags |= QtCore.Qt.ItemFlag.ItemIsDropEnabled
        self.setFlags(flags)

    
    def valueChanged(self, param, val):
        ## called when the parameter's value has changed
        pass
    
    def isFocusable(self):
        """Return True if this item should be included in the tab-focus order"""
        return False
        
    def setFocus(self):
        """Give input focus to this item.
        Can be reimplemented to display editor widgets, etc.
        """
        pass
    
    def focusNext(self, forward=True):
        """Give focus to the next (or previous) focusable item in the parameter tree"""
        self.treeWidget().focusNext(self, forward=forward)
        
    
    def treeWidgetChanged(self):
        """Called when this item is added or removed from a tree.
        Expansion, visibility, and column widgets must all be configured AFTER 
        the item is added to a tree, not during __init__.
        """
        self.setHidden(not self.param.opts.get('visible', True))
        self.setExpanded(self.param.opts.get('expanded', True))
        
    def childAdded(self, param, child, pos):
        item = child.makeTreeItem(depth=self.depth+1)
        self.insertChild(pos, item)
        item.treeWidgetChanged()
        
        for i, ch in enumerate(child):
            item.childAdded(child, ch, i)
        
    def childRemoved(self, param, child):
        for i in range(self.childCount()):
            item = self.child(i)
            if item.param is child:
                self.takeChild(i)
                break
                
    def parentChanged(self, param, parent):
        ## called when the parameter's parent has changed.
        pass
                
    def contextMenuEvent(self, ev):
        opts = self.param.opts
        
        if not opts.get('removable', False) and not opts.get('renamable', False)\
                and "context" not in opts:
            return
        
        ## Generate context menu for renaming/removing parameter
        self.contextMenu = QtWidgets.QMenu() # Put in global name space to prevent garbage collection
        self.contextMenu.addSeparator()
        self._buildParamMenu(self.contextMenu)
        self.contextMenu.popup(ev.globalPos())

    def _buildParamMenu(self, menu, show_rename=None, show_remove=None):
        """Add rename/remove/context actions to *menu*.

        *show_rename* and *show_remove* default to the ``renamable`` /
        ``removable`` parameter options when not provided, which is the
        behaviour used by the standalone right-click context menu.
        ``populateCtrlMenu`` passes explicit values so that ``ctrlActions``
        can also govern these entries.
        """
        opts = self.param.opts

        if show_rename is None:
            show_rename = opts.get('renamable', False)
        if show_remove is None:
            show_remove = opts.get('removable', False)

        if show_rename:
            act = menu.addAction(icons.getGraphIcon('rename'), translate("ParameterItem", 'Rename'))
            act.triggered.connect(self.editName)
        if show_remove:
            act = menu.addAction(icons.getGraphIcon('delete'), translate("ParameterItem", "Remove"))
            act.triggered.connect(self.requestRemove)

        context = opts.get('context', None)
        if isinstance(context, list):
            for name in context:
                menu.addAction(name).triggered.connect(self.contextMenuTriggered(name))
        elif isinstance(context, dict):
            for name, title in context.items():
                menu.addAction(title).triggered.connect(self.contextMenuTriggered(name))

    # ── Ctrl button ───────────────────────────────────────────────────────────

    @property
    def defaultBtn(self):
        """Backward-compatible alias for :attr:`ctrlBtn`."""
        return getattr(self, 'ctrlBtn', None)

    @defaultBtn.setter
    def defaultBtn(self, value):
        self.ctrlBtn = value

    def makeCtrlButton(self):
        """Create and return the ctrl :class:`CtrlButton`.

        Also sets ``self.ctrlBtn`` (and the ``self.defaultBtn`` alias) and
        ``self.ctrlMenu`` for use in :meth:`populateCtrlMenu`.
        Call once during ``__init__`` and add the returned widget to the layout.
        """
        btn = CtrlButton(self)
        self.ctrlMenu = btn._menu
        self.ctrlBtn = btn
        return btn

    def makeDefaultButton(self):
        """Deprecated. Use :meth:`makeCtrlButton` instead."""
        warnings.warn(
            "makeDefaultButton is deprecated; use makeCtrlButton instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.makeCtrlButton()

    def populateCtrlMenu(self):
        """Rebuild the ctrl button menu to reflect current parameter state.

        Override in a subclass to add custom entries; call
        ``super().populateCtrlMenu()`` to retain the built-in actions.

        Which built-in actions appear is controlled by the ``ctrlActions``
        parameter option (a set of strings, default :data:`DEFAULT_CTRL_ACTIONS`).
        Valid values are ``'default'``, ``'setDefault'``, ``'enabled'``,
        ``'readonly'``, ``'rename'``, and ``'remove'``.  For ``'rename'`` and
        ``'remove'``, either including the key in ``ctrlActions`` **or** setting
        ``renamable=True`` / ``removable=True`` on the parameter is sufficient —
        both are treated as equivalent.  ``'context'`` actions always follow the
        ``context`` parameter option and are not controlled by ``ctrlActions``.
        """
        self.ctrlMenu.clear()
        ctrl = self.param.opts.get('ctrlActions', DEFAULT_CTRL_ACTIONS)
        readonly = self.param.readonly()
        enabled = self.param.opts.get('enabled', True)

        # ── Value ─────────────────────────────────────────────────────────────
        value_count = 0 # Used to decide whether to add separators
        self._defaultAct = None
        if 'default' in ctrl and self.param.hasDefault() and not readonly:
            self._defaultAct = self.ctrlMenu.addAction(
                icons.getGraphIcon('revert_default'),
                translate("ParameterItem", "Reset to default"),
            )
            self._defaultAct.setEnabled(self.param.valueModifiedSinceResetToDefault() and enabled)
            self._defaultAct.triggered.connect(self.defaultClicked)
            value_count += 1

        if 'setDefault' in ctrl and not readonly:
            act = self.ctrlMenu.addAction(
                icons.getGraphIcon('set_default'),
                translate("ParameterItem", "Set as default"),
            )
            act.setEnabled(not self.param.valueIsDefault())
            act.triggered.connect(self._setAsDefault)
            value_count += 1

        # ── State ──────────────────────────────────────────────────────────────
        self._enabledAct = None
        self._readonlyAct = None
        state_count = 0 # Used to decide whether to add separators
        if 'enabled' in ctrl or 'readonly' in ctrl:
            if value_count:
                self.ctrlMenu.addSeparator()

        if 'enabled' in ctrl:
            self._enabledAct = self.ctrlMenu.addAction(
                icons.getGraphIcon('visibleEye') if enabled else icons.getGraphIcon('invisibleEye'),
                translate("ParameterItem", "Disable") if enabled
                else translate("ParameterItem", "Enable"),
            )
            self._enabledAct.setProperty("persistentMenu", True)
            self._enabledAct.triggered.connect(self._toggleEnabled)
            state_count += 1

        if 'readonly' in ctrl:
            self._readonlyAct = self.ctrlMenu.addAction(
                icons.getGraphIcon('lock') if not readonly else icons.getGraphIcon('unlock'),
                translate("ParameterItem", "Lock") if not readonly
                else translate("ParameterItem", "Unlock"),
            )
            self._readonlyAct.setProperty("persistentMenu", True)
            self._readonlyAct.triggered.connect(self._toggleReadonly)
            state_count += 1

        # ── Rename / Remove / Context ───────────────────────────────────────────
        # Either the parameter opt OR the presence of the key in ctrlActions
        # is sufficient to show the action.
        opts = self.param.opts
        show_rename = bool(opts.get('renamable') or 'rename' in ctrl)
        show_remove = bool(opts.get('removable') or 'remove' in ctrl)
        has_manage = show_rename or show_remove or 'context' in opts
        if has_manage:
            if value_count or state_count:
                self.ctrlMenu.addSeparator()
            self._buildParamMenu(self.ctrlMenu, show_rename, show_remove)

    def updateCtrlButton(self):
        """Refresh the ctrl button menu to reflect current parameter state.

        Called automatically on value and opts changes. Override in a subclass
        to add custom refresh logic; call ``super().updateCtrlButton()`` to
        retain the built-in menu refresh.

        When the menu is currently visible (e.g. a persistent action was just
        triggered), the rebuild is skipped — individual toggle actions update
        their own icon and text in-place, and ``aboutToShow`` ensures a full
        refresh the next time the menu opens.
        """
        if hasattr(self, 'ctrlMenu') and not self.ctrlMenu.isVisible():
            self.populateCtrlMenu()

    def updateDefaultBtn(self):
        """Deprecated. Use :meth:`updateCtrlButton` instead."""
        warnings.warn(
            "updateDefaultBtn is deprecated; use updateCtrlButton instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.updateCtrlButton()

    def defaultClicked(self):
        self.param.setToDefault()

    def _setAsDefault(self):
        self.param.setDefault(self.param.value())

    def _toggleEnabled(self):
        new_enabled = not self.param.opts.get('enabled', True)
        self.param.setOpts(enabled=new_enabled)
        if self._enabledAct is not None:
            self._enabledAct.setIcon(
                icons.getGraphIcon('visibleEye') if new_enabled
                else icons.getGraphIcon('invisibleEye')
            )
            self._enabledAct.setText(
                translate("ParameterItem", "Disable") if new_enabled
                else translate("ParameterItem", "Enable")
            )

    def _toggleReadonly(self):
        new_readonly = not self.param.readonly()
        self.param.setOpts(readonly=new_readonly)
        if self._readonlyAct is not None:
            self._readonlyAct.setIcon(
                icons.getGraphIcon('lock') if not new_readonly
                else icons.getGraphIcon('unlock')
            )
            self._readonlyAct.setText(
                translate("ParameterItem", "Lock") if not new_readonly
                else translate("ParameterItem", "Unlock")
            )

    # ── Standard item methods ─────────────────────────────────────────────────

    def columnChangedEvent(self, col):
        """Called when the text in a column has been edited (or otherwise changed).
        By default, we only use changes to column 0 to rename the parameter.
        """
        if col == 0  and (self.param.opts.get('title', None) is None):
            if self.ignoreNameColumnChange:
                return
            try:
                newName = self.param.setName(self.text(col))
            except Exception:
                self.setText(0, self.param.name())
                raise
                
            try:
                self.ignoreNameColumnChange = True
                self.nameChanged(self, newName)  ## If the parameter rejects the name change, we need to set it back.
            finally:
                self.ignoreNameColumnChange = False

    def expandedChangedEvent(self, expanded):
        if self.param.opts['syncExpanded']:
            self.param.setOpts(expanded=expanded)
                
    def nameChanged(self, param, name):
        ## called when the parameter's name has changed.
        if self.param.opts.get('title', None) is None:
            self.titleChanged()

    def titleChanged(self):
        # called when the user-visble title has changed (either opts['title'], or name if title is None)

        title = self.param.title()
        # This makes sure that items without a title or the title 'params' remain invisible
        if not title or title == 'params':
            return
        self.setText(0, title)
        fm = QtGui.QFontMetrics(self.font(0))
        textFlags = QtCore.Qt.TextFlag.TextSingleLine
        size = fm.size(textFlags, self.text(0))
        size.setHeight(int(size.height() * 1.35))
        size.setWidth(int(size.width() * 1.15))
        self.setSizeHint(0, size)

    def limitsChanged(self, param, limits):
        """Called when the parameter's limits have changed"""
        pass
    
    def defaultChanged(self, param, default):
        """Called when the parameter's default value has changed"""
        pass

    def optsChanged(self, param, opts):
        """Called when any options are changed that are not
        name, value, default, or limits"""
        if 'visible' in opts:
            self.setHidden(not opts['visible'])

        if 'expanded' in opts:
            if self.isExpanded() != opts['expanded']:
                self.setExpanded(opts['expanded'])

        if 'title' in opts:
            self.titleChanged()

        self.updateFlags()

    def contextMenuTriggered(self, name):
        def trigger():
            self.param.contextMenu(name)
        return trigger

    def editName(self):
        self.treeWidget().editItem(self, 0)
        
    def selected(self, sel):
        """Called when this item has been selected (sel=True) OR deselected (sel=False)"""
        pass

    def requestRemove(self):
        ## called when remove is selected from the context menu.
        ## we need to delay removal until the action is complete
        ## since destroying the menu in mid-action will cause a crash.
        QtCore.QTimer.singleShot(0, self.param.remove)

    ## for python 3 support, we need to redefine hash and eq methods.
    def __hash__(self):
        return id(self)

    def __eq__(self, x):
        return x is self
