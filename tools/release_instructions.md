PyQtGraph Release Procedure
---------------------------

1. Create a release-x.x.x branch

2. Run build-pg-release script
    - creates clone of master
    - merges release branch into master
    - updates version numbers in code
    - creates pyqtgraph-x.x.x tag
    - creates release commit
    - builds source dist
    - test pip install
    - builds windows dists
    - builds deb dist

3. test build files
    - test setup.py, pip on OSX
    - test setup.py, pip, 32/64 exe on windows
    - test setup.py, pip, deb on linux (py2, py3)
    
4. Run upload-release script
    - pip upload
    - github push + release
    - website upload

5. publish
    - update website
    - mailing list announcement
    - new conda recipe  (http://conda.pydata.org/docs/build.html)
    - contact various package maintainers
