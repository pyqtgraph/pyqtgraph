import collections
import importlib.util
import re

from .. import colormap
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['ColorMapMenu']


# from https://matplotlib.org/stable/gallery/color/colormap_reference.html
MATPLOTLIB_CMAPS = [
         ('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])
    ]


PrivateActionData = collections.namedtuple("ColorMapMenuPrivateActionData", ["name", "source"])


def buildMenuEntryWidget(cmap, text):
    lut = cmap.getLookupTable(nPts=32, alpha=True)
    qimg = QtGui.QImage(lut, len(lut), 1, QtGui.QImage.Format.Format_RGBA8888)
    pixmap = QtGui.QPixmap.fromImage(qimg)

    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(1,1,1,1)
    label1 = QtWidgets.QLabel()
    label1.setScaledContents(True)
    label1.setPixmap(pixmap)
    label2 = QtWidgets.QLabel(text)
    layout.addWidget(label1, 0)
    layout.addWidget(label2, 1)

    return widget

def buildMenuEntryAction(menu, name, source):
    if isinstance(source, colormap.ColorMap):
        cmap = source
    elif source == 'preset-gradient':
        cmap = preset_gradient_to_colormap(name)
    else:
        cmap = colormap.get(name, source=source)
    act = QtWidgets.QWidgetAction(menu)
    act.setData(PrivateActionData(name, source))
    act.setDefaultWidget(buildMenuEntryWidget(cmap, name))
    menu.addAction(act)

def sorted_filenames(names):
    pattern = re.compile(r'(\d+)')
    key = lambda x: [int(c) if c.isdigit() else c for c in pattern.split(x)]
    return sorted(names, key=key)

def find_mpl_leftovers():
    names = colormap.listMaps(source="matplotlib")
    # remove entries registered by colorcet
    names = [x for x in names if not x.startswith('cet_')]
    # remove the reversed colormaps
    names = [x for x in names if not x.endswith('_r')]
    # remove entries that are already categorised
    known_names = set()
    for item in MATPLOTLIB_CMAPS:
        known_names.update(item[1])
    names = [x for x in names if x not in known_names]
    return names

def buildCetSubMenu(menu, source, cet_type):
    names = colormap.listMaps(source=source)
    names = [x for x in names if x.startswith("CET")]

    if cet_type.endswith("Blind"):
        names = [x for x in names if x[4:6] == "CB"]
    else:
        names = [x for x in names if x[4] == cet_type[0] and x[5].isdigit()]

    for name in sorted_filenames(names):
        buildMenuEntryAction(menu, name, source)

def buildUserSubMenu(menu, userList):
    for item in userList:
        if isinstance(item, colormap.ColorMap):
            name, source = item.name, item
        elif isinstance(item, str):
            name, source = item, None
        elif isinstance(item, tuple):
            name, source = item
        else:
            raise ValueError("userList items must be ColorMap, str or tuple")

        buildMenuEntryAction(menu, name, source)

def preset_gradient_to_colormap(name):
    # generate the hsv two gradients using makeHslCycle
    if name == 'spectrum':
        # steps=30 for 300 degrees gives the same density as
        # the default steps=36 for 360 degrees
        cmap = colormap.makeHslCycle((0, 300/360), steps=30)
    elif name == 'cyclic':
        cmap = colormap.makeHslCycle((1, 0))
    else:
        cmap = colormap.ColorMap(*zip(*Gradients[name]["ticks"]), name=name)
    return cmap


class ColorMapMenu(QtWidgets.QMenu):
    sigColorMapTriggered = QtCore.Signal(object)

    def __init__(self, *, userList=None, showGradientSubMenu=False, showColorMapSubMenus=False):
        """
        Creates a new ColorMapMenu.

        Parameters
        ----------
        userList : list of ColorMapSpecifier, optional
            Supported values for ColorMapSpecifier are 
            ``str``, ``(str, str)``, :class:`~pyqtgraph.ColorMap`
            
            Example: ``["viridis", ("glasbey", "colorcet"), ("rainbow", "matplotlib")]``
        showGradientSubMenu : bool, default=False
            Adds legacy gradients in a submenu.
        showColorMapSubMenus : bool, default=False
            Adds bundled colormaps and external (colorcet, matplotlib) colormaps in submenus.
        """
        super().__init__()

        self.setTitle("ColorMaps")
        self.triggered.connect(self.onTriggered)

        topmenu = self
        act = topmenu.addAction('None')
        act.setData(PrivateActionData(None, None))

        if userList is not None:
            buildUserSubMenu(topmenu, userList)

        if any([showGradientSubMenu, showColorMapSubMenus]):
            topmenu.addSeparator()

        # render the submenus only if the user actually clicks on it

        if showGradientSubMenu:
            submenu = topmenu.addMenu('preset gradient')
            submenu.aboutToShow.connect(self.buildGradientSubMenu)

        if not showColorMapSubMenus:
            return

        submenu = topmenu.addMenu('local')
        submenu.aboutToShow.connect(self.buildLocalSubMenu)

        have_colorcet = importlib.util.find_spec('colorcet') is not None
        # arranged in the order listed in https://colorcet.com/
        cet_types = ["Linear", "Divergent", "Rainbow", "Cyclic", "Isoluminant", "Color Blind"]

        # the local cet files are a subset of the colorcet module.
        # expose just one of them.
        if not have_colorcet:
            submenu = topmenu.addMenu('cet (local)')
            for cet_type in cet_types:
                sub2menu = submenu.addMenu(cet_type)
                sub2menu.aboutToShow.connect(self.buildCetLocalSubMenu)
        else:
            submenu = topmenu.addMenu('cet (external)')
            for cet_type in cet_types:
                sub2menu = submenu.addMenu(cet_type)
                sub2menu.aboutToShow.connect(self.buildCetExternalSubMenu)

        if importlib.util.find_spec('matplotlib') is not None:
            submenu = topmenu.addMenu('matplotlib')
            # skip 1st entry which is "Perceptually Uniform Sequential"
            # since pyqtgraph has those already
            for category, _ in MATPLOTLIB_CMAPS[1:]:
                sub2menu = submenu.addMenu(category)
                sub2menu.aboutToShow.connect(self.buildMplCategorySubMenu)

            if find_mpl_leftovers():
                sub2menu = submenu.addMenu("Others")
                sub2menu.aboutToShow.connect(self.buildMplOthersSubMenu)

        if have_colorcet:
            submenu = topmenu.addMenu('colorcet')
            submenu.aboutToShow.connect(self.buildColorcetSubMenu)

    def onTriggered(self, action):
        if not isinstance(data := action.data(), PrivateActionData):
            return
        cmap = self.actionDataToColorMap(data)
        self.sigColorMapTriggered.emit(cmap)

    def buildGradientSubMenu(self):
        source = 'preset-gradient'
        names = list(Gradients.keys())
        self.buildSubMenu(names, source, sort=False)

    def buildLocalSubMenu(self):
        source = None
        names = colormap.listMaps(source=source)
        names = [x for x in names if not x.startswith('CET')]
        names = [x for x in names if not x.startswith('PAL-relaxed')]
        self.buildSubMenu(names, source)

    def buildCetLocalSubMenu(self):
        # in Qt6 we could have used Qt.ConnectionType.SingleShotConnection
        menu = self.sender()
        menu.aboutToShow.disconnect()
        source = None
        cet_type = menu.title()
        buildCetSubMenu(menu, source, cet_type)

    def buildCetExternalSubMenu(self):
        # in Qt6 we could have used Qt.ConnectionType.SingleShotConnection
        menu = self.sender()
        menu.aboutToShow.disconnect()
        source = 'colorcet'
        cet_type = menu.title()
        buildCetSubMenu(menu, source, cet_type)

    def buildMplCategorySubMenu(self):
        # in Qt6 we could have used Qt.ConnectionType.SingleShotConnection
        menu = self.sender()
        menu.aboutToShow.disconnect()
        source = 'matplotlib'
        category = menu.title()
        categories = [x[0] for x in MATPLOTLIB_CMAPS]
        names = MATPLOTLIB_CMAPS[categories.index(category)][1]
        for name in names:
            try:
                buildMenuEntryAction(menu, name, source)
            except ValueError:
                # the names are not programmatically discovered,
                # so to be safe, we wrap around try except
                pass

    def buildMplOthersSubMenu(self):
        self.buildSubMenu(find_mpl_leftovers(), "matplotlib")

    def buildColorcetSubMenu(self):
        # colorcet colormaps with shorter/simpler aliases
        source = 'colorcet'
        import colorcet
        names = list(colorcet.palette_n.keys())
        self.buildSubMenu(names, source)

    def buildSubMenu(self, names, source, sort=True):
        # in Qt6 we could have used Qt.ConnectionType.SingleShotConnection
        menu = self.sender()
        menu.aboutToShow.disconnect()

        if sort:
            names = sorted_filenames(names)
        for name in names:
            buildMenuEntryAction(menu, name, source)

    @staticmethod
    def actionDataToColorMap(data):
        name, source = data
        if isinstance(source, colormap.ColorMap):
            cmap = source
        elif name is None:
            cmap = colormap.ColorMap(None, [0.0, 1.0])
        elif source == 'preset-gradient':
            cmap = preset_gradient_to_colormap(name)
            cmap.name = f"{source}:{name}"    # for GradientEditorItem
        else:
            # colormap module maintains a cache keyed by name only.
            # thus if a colormap has the same name in two different sources,
            # we will end up getting whatever was already cached.
            cmap = colormap.get(name, source=source)
        return cmap
