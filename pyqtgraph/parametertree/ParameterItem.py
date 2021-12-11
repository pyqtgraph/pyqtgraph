from ..Qt import QtCore, QtGui, QtWidgets

translate = QtCore.QCoreApplication.translate

class ParameterItem(QtWidgets.QTreeWidgetItem):
    """
    Abstract ParameterTree item. 
    Used to represent the state of a Parameter from within a ParameterTree.
    
      - Sets first column of item to name
      - generates context menu if item is renamable or removable
      - handles child added / removed events
      - provides virtual functions for handling changes from parameter
    
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
        
        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        if opts.get('renamable', False):
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
        if opts.get('renamable', False):
            self.contextMenu.addAction(translate("ParameterItem", 'Rename')).triggered.connect(self.editName)
        if opts.get('removable', False):
            self.contextMenu.addAction(translate("ParameterItem", "Remove")).triggered.connect(self.requestRemove)
        
        # context menu
        context = opts.get('context', None)
        if isinstance(context, list):
            for name in context:
                self.contextMenu.addAction(name).triggered.connect(
                    self.contextMenuTriggered(name))
        elif isinstance(context, dict):
            for name, title in context.items():
                self.contextMenu.addAction(title).triggered.connect(
                    self.contextMenuTriggered(name))
        
        self.contextMenu.popup(ev.globalPos())
        
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
