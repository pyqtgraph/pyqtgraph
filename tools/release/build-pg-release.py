#!/usr/bin/python
import os, sys, argparse, random
from shell import shell, ssh



description="Build release packages for pyqtgraph."

epilog = """
Package build is done in several steps:

    * Attempt to clone branch release-x.y.z from %s
    * Merge release branch into master
    * Write new version numbers into the source
    * Roll over unreleased CHANGELOG entries
    * Commit and tag new release
    * Build HTML documentation
    * Build source package
    * Build deb packages (if running on Linux)
    * Build Windows exe installers

Building source packages requires:

    * 
    * 
    * python-sphinx

Building deb packages requires several dependencies:

    * build-essential
    * python-all, python3-all
    * python-stdeb, python3-stdeb
    
"""

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
build_dir = os.path.join(path, 'release-build')
pkg_dir = os.path.join(path, 'release-packages')

ap = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('version', help='The x.y.z version to generate release packages for. '
                                'There must be a corresponding pyqtgraph-x.y.z branch in the source repository.')
ap.add_argument('--source-repo', metavar='', help='Repository from which release and master branches will be cloned. Default is the repo containing this script.', default=path)
ap.add_argument('--build-dir', metavar='', help='Directory where packages will be staged and built. Default is source_root/release-build.', default=build_dir)
ap.add_argument('--pkg-dir', metavar='', help='Directory where packages will be stored. Default is source_root/release-packages.', default=pkg_dir)
ap.add_argument('--skip-pip-test', metavar='', help='Skip testing pip install.', action='store_const', const=True, default=False)
ap.add_argument('--no-deb', metavar='', help='Skip building Debian packages.', action='store_const', const=True, default=False)
ap.add_argument('--no-exe', metavar='', help='Skip building Windows exe installers.', action='store_const', const=True, default=False)

args = ap.parse_args()
args.build_dir = os.path.abspath(args.build_dir)
args.pkg_dir = os.path.join(os.path.abspath(args.pkg_dir), args.version)


if os.path.exists(args.build_dir):
    sys.stderr.write("Please remove the build directory %s before proceeding, or specify a different path with --build-dir.\n" % args.build_dir)
    sys.exit(-1)
if os.path.exists(args.pkg_dir):
    sys.stderr.write("Please remove the package directory %s before proceeding, or specify a different path with --pkg-dir.\n" % args.pkg_dir)
    sys.exit(-1)
    
    
version = args.version

vars = {
    'ver': args.version,
    'bld': args.build_dir,
    'src': args.source_repo,
    'pkgdir': args.pkg_dir,
}


# Clone source repository and tag the release branch
shell('''
    # Clone and merge release branch into previous master
    mkdir -p {bld}
    cd {bld}
    rm -rf pyqtgraph
    git clone --depth 1 -b master {src} pyqtgraph
    cd pyqtgraph
    git checkout -b release-{ver}
    git pull {src} release-{ver}
    git checkout master
    git merge --no-ff --no-commit release-{ver}
    
    # Write new version number into the source
    sed -i "s/__version__ = .*/__version__ = '{ver}'/" pyqtgraph/__init__.py
    sed -i "s/version = .*/version = '{ver}'/" doc/source/conf.py
    sed -i "s/release = .*/release = '{ver}'/" doc/source/conf.py
    
    # make sure changelog mentions unreleased changes
    grep "pyqtgraph-{ver}.*unreleased.*" CHANGELOG    
    sed -i "s/pyqtgraph-{ver}.*unreleased.*/pyqtgraph-{ver}/" CHANGELOG

    # Commit and tag new release
    git commit -a -m "PyQtGraph release {ver}"
    git tag pyqtgraph-{ver}

    # Build HTML documentation
    cd doc
        make clean
        make html
    cd ..
    find ./ -name "*.pyc" -delete

    # package source distribution
    python setup.py sdist

    mkdir -p {pkgdir}
    cp dist/*.tar.gz {pkgdir}

    # source package build complete.
'''.format(**vars))

    
if args.skip_pip_test:
    vars['pip_test'] = 'skipped'
else:
    shell('''
        # test pip install source distribution
        rm -rf release-{ver}-virtenv
        virtualenv --system-site-packages release-{ver}-virtenv
        . release-{ver}-virtenv/bin/activate
        echo "PATH: $PATH"
        echo "ENV: $VIRTUAL_ENV" 
        pip install --no-index --no-deps dist/pyqtgraph-{ver}.tar.gz
        deactivate
        
        # pip install test passed
    '''.format(**vars))
    vars['pip_test'] = 'passed'


if 'linux' in sys.platform and not args.no_deb: 
    shell('''
        # build deb packages
        cd {bld}/pyqtgraph
        python setup.py --command-packages=stdeb.command sdist_dsc
        cd deb_dist/pyqtgraph-{ver}
        sed -i "s/^Depends:.*/Depends: python (>= 2.6), python-qt4 | python-pyside, python-numpy/" debian/control    
        dpkg-buildpackage
        cd ../../
        mv deb_dist {pkgdir}/pyqtgraph-{ver}-deb
        
        # deb package build complete.
    '''.format(**vars))
    vars['deb_status'] = 'built'
else:
    vars['deb_status'] = 'skipped'
    

if not args.no_exe:
    shell("""
        # Build windows executables
        cd {bld}/pyqtgraph
        python setup.py build bdist_wininst --plat-name=win32
        python setup.py build bdist_wininst
        cp dist/*.exe {pkgdir}
    """.format(**vars))
    vars['exe_status'] = 'built'    
else:
    vars['exe_status'] = 'skipped'


print("""

======== Build complete. =========

* Source package:     built
* Pip install test:   {pip_test}
* Debian packages:    {deb_status}
* Windows installers: {exe_status}
* Package files in    {pkgdir}
""".format(**vars))
