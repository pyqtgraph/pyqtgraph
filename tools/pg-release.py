#!/usr/bin/python
import os, sys, argparse, random
from shell import shell, ssh



description="Build release packages for pyqtgraph."

epilog = """
Package build is done in several steps:

    * Attempt to clone branch release-x.y.z from source-repo
    * Merge release branch into master
    * Write new version numbers into the source
    * Roll over unreleased CHANGELOG entries
    * Commit and tag new release
    * Build HTML documentation
    * Build source package
    * Build deb packages (if running on Linux)
    * Build Windows exe installers

Release packages may be published by using the --publish flag:

    * Uploads release files to website
    * Pushes tagged git commit to github
    * Uploads source package to pypi

Building source packages requires:

    * 
    * 
    * python-sphinx

Building deb packages requires several dependencies:

    * build-essential
    * python-all, python3-all
    * python-stdeb, python3-stdeb
    
Note: building windows .exe files should be possible on any OS. However, 
Debian/Ubuntu systems do not include the necessary wininst*.exe files; these
must be manually copied from the Python source to the distutils/command 
submodule path (/usr/lib/pythonX.X/distutils/command). Additionally, it may be
necessary to rename (or copy / link) wininst-9.0-amd64.exe to 
wininst-6.0-amd64.exe.

"""

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_dir = os.path.join(path, 'release-build')
pkg_dir = os.path.join(path, 'release-packages')

ap = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('version', help='The x.y.z version to generate release packages for. '
                                'There must be a corresponding pyqtgraph-x.y.z branch in the source repository.')
ap.add_argument('--publish', metavar='', help='Publish previously built package files (must be stored in pkg-dir/version) and tagged release commit (from build-dir).', action='store_const', const=True, default=False)
ap.add_argument('--source-repo', metavar='', help='Repository from which release and master branches will be cloned. Default is the repo containing this script.', default=path)
ap.add_argument('--build-dir', metavar='', help='Directory where packages will be staged and built. Default is source_root/release-build.', default=build_dir)
ap.add_argument('--pkg-dir', metavar='', help='Directory where packages will be stored. Default is source_root/release-packages.', default=pkg_dir)
ap.add_argument('--skip-pip-test', metavar='', help='Skip testing pip install.', action='store_const', const=True, default=False)
ap.add_argument('--no-deb', metavar='', help='Skip building Debian packages.', action='store_const', const=True, default=False)
ap.add_argument('--no-exe', metavar='', help='Skip building Windows exe installers.', action='store_const', const=True, default=False)



def build(args):
    if os.path.exists(args.build_dir):
        sys.stderr.write("Please remove the build directory %s before proceeding, or specify a different path with --build-dir.\n" % args.build_dir)
        sys.exit(-1)
    if os.path.exists(args.pkg_dir):
        sys.stderr.write("Please remove the package directory %s before proceeding, or specify a different path with --pkg-dir.\n" % args.pkg_dir)
        sys.exit(-1)
        
    # Clone source repository and tag the release branch
    shell('''
        # Clone and merge release branch into previous master
        mkdir -p {build_dir}
        cd {build_dir}
        rm -rf pyqtgraph
        git clone --depth 1 -b master {source_repo} pyqtgraph
        cd pyqtgraph
        git checkout -b release-{version}
        git pull {source_repo} release-{version}
        git checkout master
        git merge --no-ff --no-commit release-{version}
        
        # Write new version number into the source
        sed -i "s/__version__ = .*/__version__ = '{version}'/" pyqtgraph/__init__.py
        sed -i "s/version = .*/version = '{version}'/" doc/source/conf.py
        sed -i "s/release = .*/release = '{version}'/" doc/source/conf.py
        
        # make sure changelog mentions unreleased changes
        grep "pyqtgraph-{version}.*unreleased.*" CHANGELOG    
        sed -i "s/pyqtgraph-{version}.*unreleased.*/pyqtgraph-{version}/" CHANGELOG

        # Commit and tag new release
        git commit -a -m "PyQtGraph release {version}"
        git tag pyqtgraph-{version}

        # Build HTML documentation
        cd doc
            make clean
            make html
        cd ..
        find ./ -name "*.pyc" -delete

        # package source distribution
        python setup.py sdist

        mkdir -p {pkg_dir}
        cp dist/*.tar.gz {pkg_dir}

        # source package build complete.
    '''.format(**args.__dict__))

        
    if args.skip_pip_test:
        args.pip_test = 'skipped'
    else:
        shell('''
            # test pip install source distribution
            rm -rf release-{version}-virtenv
            virtualenv --system-site-packages release-{version}-virtenv
            . release-{version}-virtenv/bin/activate
            echo "PATH: $PATH"
            echo "ENV: $VIRTUAL_ENV" 
            pip install --no-index --no-deps dist/pyqtgraph-{version}.tar.gz
            deactivate
            
            # pip install test passed
        '''.format(**args.__dict__))
        args.pip_test = 'passed'


    if 'linux' in sys.platform and not args.no_deb: 
        shell('''
            # build deb packages
            cd {build_dir}/pyqtgraph
            python setup.py --command-packages=stdeb.command sdist_dsc
            cd deb_dist/pyqtgraph-{version}
            sed -i "s/^Depends:.*/Depends: python (>= 2.6), python-qt4 | python-pyside, python-numpy/" debian/control    
            dpkg-buildpackage
            cd ../../
            mv deb_dist {pkg_dir}/pyqtgraph-{version}-deb
            
            # deb package build complete.
        '''.format(**args.__dict__))
        args.deb_status = 'built'
    else:
        args.deb_status = 'skipped'
        

    if not args.no_exe:
        shell("""
            # Build windows executables
            cd {build_dir}/pyqtgraph
            python setup.py build bdist_wininst --plat-name=win32
            python setup.py build bdist_wininst --plat-name=win-amd64
            cp dist/*.exe {pkg_dir}
        """.format(**args.__dict__))
        args.exe_status = 'built'    
    else:
        args.exe_status = 'skipped'


    print(unindent("""

    ======== Build complete. =========

    * Source package:     built
    * Pip install test:   {pip_test}
    * Debian packages:    {deb_status}
    * Windows installers: {exe_status}
    * Package files in    {pkg_dir}

    Next steps to publish:
    
    * Test all packages
    * Run script again with --publish

    """).format(**args.__dict__))


def publish(args):


    if not os.path.isfile(os.path.expanduser('~/.pypirc')):
        print(unindent("""
            Missing ~/.pypirc file. Should look like:
            -----------------------------------------

                [distutils]
                index-servers =
                    pypi

                [pypi]
                username:your_username
                password:your_password

        """))
        sys.exit(-1)

    ### Upload everything to server
    shell("""
        # Uploading documentation..
        cd {build_dir}/pyqtgraph
        rsync -rv doc/build/* pyqtgraph.org:/www/code/pyqtgraph/pyqtgraph/documentation/build/

        # Uploading release packages to website
        rsync -v {pkg_dir}/{version} pyqtgraph.org:/www/code/pyqtgraph/downloads/

        # Push to github
        git push --tags https://github.com/pyqtgraph/pyqtgraph master:master

        # Upload to pypi..
        python setup.py sdist upload

    """.format(**args.__dict__))

    print(unindent("""

    ======== Upload complete. =========

    Next steps to publish:
        - update website
        - mailing list announcement
        - new conda recipe (http://conda.pydata.org/docs/build.html)
        - contact deb maintainer (gianfranco costamagna)
        - other package maintainers?

    """).format(**args.__dict__))


def unindent(msg):
    ind = 1e6
    lines = msg.split('\n')
    for line in lines:
        if len(line.strip()) == 0:
            continue
        ind = min(ind, len(line) - len(line.lstrip()))
    return '\n'.join([line[ind:] for line in lines])


if __name__ == '__main__':
    args = ap.parse_args()
    args.build_dir = os.path.abspath(args.build_dir)
    args.pkg_dir = os.path.join(os.path.abspath(args.pkg_dir), args.version)

    if args.publish:
        publish(args)
    else:
        build(args)
