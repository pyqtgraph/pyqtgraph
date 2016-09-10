PyQtGraph Release Procedure
---------------------------

0. Create your release_config.py based on release_config.example.py

1. Create a release-x.x.x branch

2. Run build-release script
    - creates clone of master from github
    - merges release branch into master
    - updates version numbers in code
    - creates pyqtgraph-x.x.x tag
    - creates release commit
    - builds source dist
    - builds windows dists
    - builds deb dist

3. test build files
    - test setup.py, pip on OSX
    - test 32/64 exe on windows
    - deb on linux (py2, py3)
    - source install on linux (py2, py3)
    
4. Run upload-release script
    - pip upload
    - github push + release
    - website upload

5. publish
    - update website
    - mailing list announcement
    - new conda recipe  (http://conda.pydata.org/docs/build.html)
    - contact deb maintainer
    - other package maintainers?
