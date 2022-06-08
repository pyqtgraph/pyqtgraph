#
# pyqtgraph documentation build configuration file, created by
# sphinx-quickstart on Fri Nov 18 19:33:12 2011.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys
import time
from datetime import datetime

import qtgallery
from sphinx_gallery.sorting import ExampleTitleSortKey

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(path, '..', '..'))
sys.path.insert(0, os.path.join(path, '..', 'extensions'))
import pyqtgraph

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'qtgallery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pyqtgraph'
now = datetime.utcfromtimestamp(
    int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
)
copyright = '2011 - {}, Luke Campagnola'.format(now.year)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = pyqtgraph.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

autodoc_inherit_docstrings = False
autodoc_mock_imports = [
    "scipy",
    "h5py",
    "matplotlib",
]

# PyQt6 seems to work the best
# C++ object deleted... msgs seem to be more warnings, renders usually look ok
runlist = [
    "Arrow",
    "BarGraphItem",
    "beeswarm",
    "CLIexample",
    "ColorBarItem",  # internal c++ object (viewbox) already deleted
    "ColorButton",
    # "ColorGradientPlots",  # not a particularly good example, need to wait to scroll
    "colorMapsLinearized",
    "colorMaps",
    "ConsoleWidget",
    "contextMenu",
    "crosshair",
    "customGraphicsItem",
    "CustomGraphItem",
    "customPlot",  # doesn't show the triangles along the x axis
    "DataSlicing",
    "DataTreeWidget",  # wrapped c/c++ object of type QComboBox has been deleted
    # "DateAxisItem",  # can't use __file__
    # "DateAxisItem_QtDesigner",  # can't use __file__
    # "designerExample",  # can't use __file__
    "DiffTreeWidget",
    "dockarea",
    # "Draw",  # necessarily interactive
    "ErrorBarItem",
    # "FillBetweenItem",  # animated
    "FlowchartCustomNode",
    "Flowchart",
    "fractal",
    "glow",

    # some GL examples do seem to work, but others give GL errors
    "GLBarGraphItem",
    "GLGradientLegendItem",
    "GLGraphItem",
    "GLImageItem",
    "GLIsosurface",
    "GLLinePlotItem",
    "GLMeshItem",
    "GLPainterItem",
    "GLScatterPlotItem",
    "GLshaders",
    "GLSurfacePlot",
    "GLTextItem",
    "GLViewWidget",
    "GLVolumeItem",

    "GradientEditor",
    "GradientWidget",
    "GraphicsLayout",
    "GraphicsScene",
    "GraphItem",
    "hdf5.py",
    "HistogramLUT",
    "histogram",
    "imageAnalysis",
    "ImageItem",
    "ImageView",
    "infiniteline_performance",
    "InfiniteLine",
    "isocurve",
    "JoystickButton",
    "Legend",
    "linkedViews",
    "logAxis",
    "LogPlotTest",
    "MatrixDisplayExample",
    "MouseSelection",
    "MultiplePlotAxes",
    "multiplePlotSpeedTest",
    "MultiPlotSpeedTest",
    "MultiPlotWidget",
    # "multiprocess",  # swallowed exception, not sure what's wrong
    "NonUniformImage",
    "optics_demos",
    # "PanningPlot",  # animated
    # "parallelize",  # hangs
    "parametertree",
    "PColorMeshItem",
    "PlotAutoRange",
    "PlotSpeedTest",
    "Plotting",
    "PlotWidget",
    # "ProgressDialog",  # hangs
    "relativity_demo",
    # "RemoteGraphicsView",  # doesn't show any plot contents
    # "RemoteSpeedTest",  # remoteproxy NoResultError
    "ROIExamples",
    "ROItypes",
    "ScaleBar",
    "ScatterPlot",
    "ScatterPlotSpeedTest",
    "ScatterPlotWidget",
    "scrollingPlots",
    "SimplePlot",
    "SpinBox",
    "Symbols",
    "TableWidget",
    "text",
    "TreeWidget",
    "verlet_chain_demo",
    "VideoSpeedTest",
    "ViewBoxFeatures",
    "ViewBox",
    "ViewLimits",
]

DEFAULT_CONFIG_OPTIONS = pyqtgraph.CONFIG_OPTIONS.copy()


def reset_pg_config(gallery_conf, fname):
    """sphinx-gallery reset callback to reset to default pg config"""
    pyqtgraph.setConfigOptions(**DEFAULT_CONFIG_OPTIONS)


def argv_handler(gallery_conf, script_vars):
    if script_vars["src_file"].endswith("hdf5.py"):
        return ["test.hdf5", "1000000"]
    else:
        return []


# sphinx gallery config
sphinx_gallery_conf = {
    "examples_dirs": os.path.join(
        os.path.dirname(__file__), "..", "..", "pyqtgraph", "examples"
    ),
    "gallery_dirs": "examples",
    "image_scrapers": (qtgallery.qtscraper,),
    "reset_modules": (qtgallery.reset_qapp, reset_pg_config),
    "filename_pattern": r"/examples/\b(" + r"|".join("{}".format(n) for n in runlist) + r")\b",
    # "fhlename_pattern": r"/examples/.*[^/]",
    "ignore_pattern": r"(/_)|(^_)|(setup)|(Template)|(template)|(test_examples)|(py2exe)",
    "within_subsection_order": ExampleTitleSortKey,
    "reset_argv": argv_handler,
    "reference_url": {
        "pyqtgraph": None,
    }
}

qtgallery_conf = {
    "xvfb_size": (1280, 800),
    "xfvb_extra_args": ["-ac", "+extension", "GLX", "+render"],
}

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# add the theme customizations
def setup(app):
    app.add_css_file("custom.css")

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyqtgraphdoc'


# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'pyqtgraph.tex', 'pyqtgraph Documentation',
   'Luke Campagnola', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'pyqtgraph', 'pyqtgraph Documentation',
     ['Luke Campagnola'], 1)
]
