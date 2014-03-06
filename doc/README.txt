
Plugins
-------------------------------
requires the sphinx plugin  "sphinxcontrib.fulltoc"
which show the TOC in the left block

-  pip install sphinxcontrib-fulltoc
or
-  easy_install sphinxcontrib-fulltoc


Style
-------------------------------
the _templates/page.html inserts some
header at top

The Font and colors are defined in
_static/pyqtgraph.css

Uses a google font loaded in page.html


Enviroment
-------------------------------

The script sets

os.environ['__GEN_DOCS__'] = "1"

within surce files use eg

if HAVE_GL and os.getenv( "__GEN_DOCS__" ):
  # do documentation
  
  
its assumed that all stuff is installed on machine inc opengl

Examples
-------------------------------
Pre compilation.. run the ./gen_files.py script

This creates the examples index (and later maybe others)

