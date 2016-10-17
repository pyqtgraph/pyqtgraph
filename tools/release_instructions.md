PyQtGraph Release Procedure
---------------------------

1. Create a release-x.x.x branch

2. Run pyqtgraph/tools/pg-release.py script (this has only been tested on linux)
    - creates clone of master
    - merges release branch into master
    - updates version numbers in code
    - creates pyqtgraph-x.x.x tag
    - creates release commit
    - builds documentation
    - builds source package
    - tests pip install
    - builds windows .exe installers (note: it may be necessary to manually
      copy wininst*.exe files from the python source packages)
    - builds deb package (note: official debian packages are built elsewhere;
      these locally-built deb packages may be phased out)

3. test build files
    - test setup.py, pip on OSX
    - test setup.py, pip, 32/64 exe on windows
    - test setup.py, pip, deb on linux (py2, py3)
    
4. Run pg-release.py script again with --publish flag
    - website upload
    - github push + release
    - pip upload

5. publish
    - update website
    - mailing list announcement
    - new conda recipe  (http://conda.pydata.org/docs/build.html)
    - contact various package maintainers
