import os, sys, re
from subprocess import check_output

def listAllPackages(pkgroot):
    path = os.getcwd()
    n = len(path.split(os.path.sep))
    subdirs = [i[0].split(os.path.sep)[n:] for i in os.walk(os.path.join(path, pkgroot)) if '__init__.py' in i[2]]
    return ['.'.join(p) for p in subdirs]


def getInitVersion(pkgroot):
    """Return the version string defined in __init__.py"""
    path = os.getcwd()
    initfile = os.path.join(path, pkgroot, '__init__.py')
    init = open(initfile).read()
    m = re.search(r'__version__ = (\S+)\n', init)
    if m is None or len(m.groups()) != 1:
        raise Exception("Cannot determine __version__ from init file: '%s'!" % initfile)
    version = m.group(1).strip('\'\"')
    return version

def gitCommit(name):
    """Return the commit ID for the given name."""
    commit = check_output(['git', 'show', name], universal_newlines=True).split('\n')[0]
    assert commit[:7] == 'commit '
    return commit[7:]

def getGitVersion(tagPrefix):
    """Return a version string with information about this git checkout.
    If the checkout is an unmodified, tagged commit, then return the tag version.
    If this is not a tagged commit, return version-branch_name-commit_id.
    If this checkout has been modified, append "+" to the version.
    """
    path = os.getcwd()
    if not os.path.isdir(os.path.join(path, '.git')):
        return None
        
    # Find last tag matching "tagPrefix.*"
    tagNames = check_output(['git', 'tag'], universal_newlines=True).strip().split('\n')
    while True:
        if len(tagNames) == 0:
            raise Exception("Could not determine last tagged version.")
        lastTagName = tagNames.pop()
        if re.match(tagPrefix+r'\d+\.\d+.*', lastTagName):
            break
    gitVersion = lastTagName.replace(tagPrefix, '')
    
    # is this commit an unchanged checkout of the last tagged version? 
    lastTag = gitCommit(lastTagName)
    head = gitCommit('HEAD')
    if head != lastTag:
        branch = re.search(r'\* (.*)', check_output(['git', 'branch'], universal_newlines=True)).group(1)
        gitVersion = gitVersion + "-%s-%s" % (branch, head[:10])
    
    # any uncommitted modifications?
    modified = False
    status = check_output(['git', 'status', '-s'], universal_newlines=True).strip().split('\n')
    for line in status:
        if line[:2] != '??':
            modified = True
            break        
                
    if modified:
        gitVersion = gitVersion + '+'

    return gitVersion

def getVersionStrings(pkg):
    """
    Returns 4 version strings: 
    
    * the version string to use for this build,
    * version string requested with --force-version (or None)
    * version string that describes the current git checkout (or None).
    * version string in the pkg/__init__.py, 
    
    The first return value is (forceVersion or gitVersion or initVersion).
    """
    
    ## Determine current version string from __init__.py
    initVersion = getInitVersion(pkgroot='pyqtgraph')

    ## If this is a git checkout, try to generate a more descriptive version string
    try:
        gitVersion = getGitVersion(tagPrefix='pyqtgraph-')
    except:
        gitVersion = None
        sys.stderr.write("This appears to be a git checkout, but an error occurred "
                        "while attempting to determine a version string for the "
                        "current commit.\n")
        sys.excepthook(*sys.exc_info())

    # See whether a --force-version flag was given
    forcedVersion = None
    for i,arg in enumerate(sys.argv):
        if arg.startswith('--force-version'):
            if arg == '--force-version':
                forcedVersion = sys.argv[i+1]
                sys.argv.pop(i)
                sys.argv.pop(i)
            elif arg.startswith('--force-version='):
                forcedVersion = sys.argv[i].replace('--force-version=', '')
                sys.argv.pop(i)
                
    ## Finally decide on a version string to use:
    if forcedVersion is not None:
        version = forcedVersion
    elif gitVersion is not None:
        version = gitVersion
        sys.stderr.write("Detected git commit; will use version string: '%s'\n" % version)
    else:
        version = initVersion

    return version, forcedVersion, gitVersion, initVersion