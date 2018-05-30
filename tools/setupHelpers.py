# -*- coding: utf-8 -*-
import os, sys, re
try:
    from subprocess import check_output, check_call
except ImportError:
    import subprocess as sp
    def check_output(*args, **kwds):
        kwds['stdout'] = sp.PIPE
        proc = sp.Popen(*args, **kwds)
        output = proc.stdout.read()
        proc.wait()
        if proc.returncode != 0:
            ex = Exception("Process had nonzero return value %d" % proc.returncode)
            ex.returncode = proc.returncode
            ex.output = output
            raise ex
        return output

# Maximum allowed repository size difference (in kB) following merge.
# This is used to prevent large files from being inappropriately added to 
# the repository history.
MERGE_SIZE_LIMIT = 100

# Paths that are checked for style by flake and flake_diff
FLAKE_CHECK_PATHS = ['pyqtgraph', 'examples', 'tools']

# Flake style checks -- mandatory, recommended, optional
# See: http://pep8.readthedocs.org/en/1.4.6/intro.html
# and  https://flake8.readthedocs.org/en/2.0/warnings.html
FLAKE_MANDATORY = set([
    'E101',  #  indentation contains mixed spaces and tabs
    'E112',  #  expected an indented block
    'E122',  #  continuation line missing indentation or outdented
    'E125',  #  continuation line does not distinguish itself from next line
    'E133',  #  closing bracket is missing indentation

    'E223',  #  tab before operator
    'E224',  #  tab after operator
    'E242',  #  tab after ‘,’
    'E273',  #  tab after keyword
    'E274',  #  tab before keyword

    'E901',  #  SyntaxError or IndentationError
    'E902',  #  IOError
        
    'W191',  #  indentation contains tabs
        
    'W601',  #  .has_key() is deprecated, use ‘in’
    'W602',  #  deprecated form of raising exception
    'W603',  #  ‘<>’ is deprecated, use ‘!=’
    'W604',  #  backticks are deprecated, use ‘repr()’    
    ])

FLAKE_RECOMMENDED = set([
    'E124',  #  closing bracket does not match visual indentation
    'E231',  #  missing whitespace after ‘,’
    
    'E211',  #  whitespace before ‘(‘
    'E261',  #  at least two spaces before inline comment
    'E271',  #  multiple spaces after keyword
    'E272',  #  multiple spaces before keyword
    'E304',  #  blank lines found after function decorator

    'F401',  #  module imported but unused
    'F402',  #  import module from line N shadowed by loop variable
    'F403',  #  ‘from module import *’ used; unable to detect undefined names
    'F404',  #  future import(s) name after other statements
        
    'E501',  #  line too long (82 > 79 characters)
    'E502',  #  the backslash is redundant between brackets
    
    'E702',  #  multiple statements on one line (semicolon)
    'E703',  #  statement ends with a semicolon
    'E711',  #  comparison to None should be ‘if cond is None:’
    'E712',  #  comparison to True should be ‘if cond is True:’ or ‘if cond:’
    'E721',  #  do not compare types, use ‘isinstance()’

    'F811',  #  redefinition of unused name from line N
    'F812',  #  list comprehension redefines name from line N
    'F821',  #  undefined name name
    'F822',  #  undefined name name in __all__
    'F823',  #  local variable name ... referenced before assignment
    'F831',  #  duplicate argument name in function definition
    'F841',  #  local variable name is assigned to but never used
    
    'W292',  #  no newline at end of file

    ])

FLAKE_OPTIONAL = set([
    'E121',  #  continuation line indentation is not a multiple of four
    'E123',  #  closing bracket does not match indentation of opening bracket
    'E126',  #  continuation line over-indented for hanging indent
    'E127',  #  continuation line over-indented for visual indent
    'E128',  #  continuation line under-indented for visual indent
        
    'E201',  #  whitespace after ‘(‘
    'E202',  #  whitespace before ‘)’
    'E203',  #  whitespace before ‘:’
    'E221',  #  multiple spaces before operator
    'E222',  #  multiple spaces after operator
    'E225',  #  missing whitespace around operator
    'E227',  #  missing whitespace around bitwise or shift operator
    'E226',  #  missing whitespace around arithmetic operator
    'E228',  #  missing whitespace around modulo operator
    'E241',  #  multiple spaces after ‘,’
    'E251',  #  unexpected spaces around keyword / parameter equals
    'E262',  #  inline comment should start with ‘# ‘     
        
    'E301',  #  expected 1 blank line, found 0
    'E302',  #  expected 2 blank lines, found 0
    'E303',  #  too many blank lines (3)
        
    'E401',  #  multiple imports on one line

    'E701',  #  multiple statements on one line (colon)
        
    'W291',  #  trailing whitespace
    'W293',  #  blank line contains whitespace
        
    'W391',  #  blank line at end of file
    ])

FLAKE_IGNORE = set([
    # 111 and 113 are ignored because they appear to be broken.
    'E111',  #  indentation is not a multiple of four
    'E113',  #  unexpected indentation
    ])


#def checkStyle():
    #try:
        #out = check_output(['flake8', '--select=%s' % FLAKE_TESTS, '--statistics', 'pyqtgraph/'])
        #ret = 0
        #print("All style checks OK.")
    #except Exception as e:
        #out = e.output
        #ret = e.returncode
        #print(out.decode('utf-8'))
    #return ret


def checkStyle():
    """ Run flake8, checking only lines that are modified since the last
    git commit. """
    test = [ 1,2,3 ]
    
    # First check _all_ code against mandatory error codes
    print('flake8: check all code against mandatory error set...')
    errors = ','.join(FLAKE_MANDATORY)
    cmd = ['flake8', '--select=' + errors] + FLAKE_CHECK_PATHS
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    #ret = proc.wait()
    output = proc.stdout.read().decode('utf-8')
    ret = proc.wait()
    printFlakeOutput(output)
    
    # Check for DOS newlines
    print('check line endings in all files...')
    count = 0
    allowedEndings = set([None, '\n'])
    for path, dirs, files in os.walk('.'):
        for f in files:
            if os.path.splitext(f)[1] not in ('.py', '.rst'):
                continue
            filename = os.path.join(path, f)
            fh = open(filename, 'U')
            x = fh.readlines()
            endings = set(fh.newlines if isinstance(fh.newlines, tuple) else (fh.newlines,))
            endings -= allowedEndings
            if len(endings) > 0:
                print("\033[0;31m" + "File has invalid line endings: %s" % filename + "\033[0m")
                ret = ret | 2
            count += 1
    print('checked line endings in %d files' % count)
            
    
    # Next check new code with optional error codes
    print('flake8: check new code against recommended error set...')
    diff = subprocess.check_output(['git', 'diff'])
    proc = subprocess.Popen(['flake8', '--diff', #'--show-source',
                                '--ignore=' + errors],
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE)
    proc.stdin.write(diff)
    proc.stdin.close()
    output = proc.stdout.read().decode('utf-8')
    ret |= printFlakeOutput(output)
    
    if ret == 0:
        print('style test passed.')
    else:
        print('style test failed: %d' % ret)
    return ret

def printFlakeOutput(text):
    """ Print flake output, colored by error category.
    Return 2 if there were any mandatory errors,
    1 if only recommended / optional errors, and
    0 if only optional errors.
    """
    ret = 0
    gotError = False
    for line in text.split('\n'):
        m = re.match(r'[^\:]+\:\d+\:\d+\: (\w+) .*', line)
        if m is None:
            print(line)
        else:
            gotError = True
            error = m.group(1)
            if error in FLAKE_MANDATORY:
                print("\033[0;31m" + line + "\033[0m")
                ret |= 2
            elif error in FLAKE_RECOMMENDED:
                print("\033[0;33m" + line + "\033[0m")
                #ret |= 1
            elif error in FLAKE_OPTIONAL:
                print("\033[0;32m" + line + "\033[0m")
            elif error in FLAKE_IGNORE:
                continue
            else:
                print("\033[0;36m" + line + "\033[0m")
    if not gotError:
        print("    [ no errors ]\n")
    return ret



def unitTests():
    """
    Run all unit tests (using py.test)
    Return the exit code.
    """
    try:
        if sys.version[0] == '3':
            out = check_output('PYTHONPATH=. py.test-3', shell=True)
        else:
            out = check_output('PYTHONPATH=. py.test', shell=True)
        ret = 0
    except Exception as e:
        out = e.output
        ret = e.returncode
    print(out.decode('utf-8'))
    return ret


def checkMergeSize(sourceBranch=None, targetBranch=None, sourceRepo=None, targetRepo=None):
    """
    Check that a git merge would not increase the repository size by MERGE_SIZE_LIMIT.
    """
    if sourceBranch is None:
        sourceBranch = getGitBranch()
        sourceRepo = '..'
        
    if targetBranch is None:
        if sourceBranch == 'develop':
            targetBranch = 'develop'
            targetRepo = 'https://github.com/pyqtgraph/pyqtgraph.git'
        else:
            targetBranch = 'develop'
            targetRepo = '..'
    
    workingDir = '__merge-test-clone'
    env = dict(TARGET_BRANCH=targetBranch, 
               SOURCE_BRANCH=sourceBranch, 
               TARGET_REPO=targetRepo, 
               SOURCE_REPO=sourceRepo,
               WORKING_DIR=workingDir,
               )
    
    print("Testing merge size difference:\n"
          "  SOURCE: {SOURCE_REPO} {SOURCE_BRANCH}\n"
          "  TARGET: {TARGET_BRANCH} {TARGET_REPO}".format(**env))
    
    setup = """
        mkdir {WORKING_DIR} && cd {WORKING_DIR} &&
        git init && git remote add -t {TARGET_BRANCH} target {TARGET_REPO} &&
        git fetch target {TARGET_BRANCH} && 
        git checkout -qf target/{TARGET_BRANCH} && 
        git gc -q --aggressive
        """.format(**env)
        
    checkSize = """
        cd {WORKING_DIR} && 
        du -s . | sed -e "s/\t.*//"
        """.format(**env)
    
    merge = """
        cd {WORKING_DIR} &&
        git pull -q {SOURCE_REPO} {SOURCE_BRANCH} && 
        git gc -q --aggressive
        """.format(**env)
    
    try:
        print("Check out target branch:\n" + setup)
        check_call(setup, shell=True)
        targetSize = int(check_output(checkSize, shell=True))
        print("TARGET SIZE: %d kB" % targetSize)
        print("Merge source branch:\n" + merge)
        check_call(merge, shell=True)
        mergeSize = int(check_output(checkSize, shell=True))
        print("MERGE SIZE: %d kB" % mergeSize)
        
        diff = mergeSize - targetSize
        if diff <= MERGE_SIZE_LIMIT:
            print("DIFFERENCE: %d kB  [OK]" % diff)
            return 0
        else:
            print("\033[0;31m" + "DIFFERENCE: %d kB  [exceeds %d kB]" % (diff, MERGE_SIZE_LIMIT) + "\033[0m")
            return 2
    finally:
        if os.path.isdir(workingDir):
            shutil.rmtree(workingDir)


def mergeTests():
    ret = checkMergeSize()
    ret |= unitTests()
    ret |= checkStyle()
    if ret == 0:
        print("\033[0;32m" + "\nAll merge tests passed." + "\033[0m")
    else:
        print("\033[0;31m" + "\nMerge tests failed." + "\033[0m")
    return ret


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
    If this is not a tagged commit, return the output of ``git describe --tags``.
    If this checkout has been modified, append "+" to the version.
    """
    path = os.getcwd()
    if not os.path.isdir(os.path.join(path, '.git')):
        return None
        
    v = check_output(['git', 'describe', '--tags', '--dirty', '--match=%s*'%tagPrefix]).strip().decode('utf-8')
    
    # chop off prefix
    assert v.startswith(tagPrefix)
    v = v[len(tagPrefix):]

    # split up version parts
    parts = v.split('-')
    
    # has working tree been modified?
    modified = False
    if parts[-1] == 'dirty':
        modified = True
        parts = parts[:-1]
        
    # have commits been added on top of last tagged version?
    # (git describe adds -NNN-gXXXXXXX if this is the case)
    local = None
    if len(parts) > 2 and re.match(r'\d+', parts[-2]) and re.match(r'g[0-9a-f]{7}', parts[-1]):
        local = parts[-1]
        parts = parts[:-2]
        
    gitVersion = '-'.join(parts)
    if local is not None:
        gitVersion += '+' + local
    if modified:
        gitVersion += 'm'

    return gitVersion

def getGitBranch():
    m = re.search(r'\* (.*)', check_output(['git', 'branch'], universal_newlines=True))
    if m is None:
        return ''
    else:
        return m.group(1)

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
    initVersion = getInitVersion(pkgroot=pkg)

    ## If this is a git checkout, try to generate a more descriptive version string
    try:
        gitVersion = getGitVersion(tagPrefix=pkg+'-')
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
    else:
        version = initVersion
        # if git says this is a modified branch, add local version information
        if gitVersion is not None:
            _, local = gitVersion.split('+')
            if local != '':
                version = version + '+' + local
                sys.stderr.write("Detected git commit; will use version string: '%s'\n" % version)

    return version, forcedVersion, gitVersion, initVersion


from distutils.core import Command
import shutil, subprocess
from generateChangelog import generateDebianChangelog

class DebCommand(Command):
    description = "build .deb package using `debuild -us -uc`"
    maintainer = "Luke Campagnola <luke.campagnola@gmail.com>"
    debTemplate = "debian"
    debDir = "deb_build"
    
    user_options = []
    
    def initialize_options(self):
        self.cwd = None
        
    def finalize_options(self):
        self.cwd = os.getcwd()
        
    def run(self):
        version = self.distribution.get_version()
        pkgName = self.distribution.get_name()
        debName = "python-" + pkgName
        debDir = self.debDir
        
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        
        if os.path.isdir(debDir):
            raise Exception('DEB build dir already exists: "%s"' % debDir)
        sdist = "dist/%s-%s.tar.gz" % (pkgName, version)
        if not os.path.isfile(sdist):
            raise Exception("No source distribution; run `setup.py sdist` first.")
        
        # copy sdist to build directory and extract
        os.mkdir(debDir)
        renamedSdist = '%s_%s.orig.tar.gz' % (debName, version)
        print("copy %s => %s" % (sdist, os.path.join(debDir, renamedSdist)))
        shutil.copy(sdist, os.path.join(debDir, renamedSdist))
        print("cd %s; tar -xzf %s" % (debDir, renamedSdist))
        if os.system("cd %s; tar -xzf %s" % (debDir, renamedSdist)) != 0:
            raise Exception("Error extracting source distribution.")
        buildDir = '%s/%s-%s' % (debDir, pkgName, version)
        
        # copy debian control structure
        print("copytree %s => %s" % (self.debTemplate, buildDir+'/debian'))
        shutil.copytree(self.debTemplate, buildDir+'/debian')
        
        # Write new changelog
        chlog = generateDebianChangelog(pkgName, 'CHANGELOG', version, self.maintainer)
        print("write changelog %s" % buildDir+'/debian/changelog')
        open(buildDir+'/debian/changelog', 'w').write(chlog)
        
        # build package
        print('cd %s; debuild -us -uc' % buildDir)
        if os.system('cd %s; debuild -us -uc' % buildDir) != 0:
            raise Exception("Error during debuild.")


class DebugCommand(Command):
    """Just for learning about distutils."""
    description = ""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        global cmd
        cmd = self
        print(self.distribution.name)
        print(self.distribution.version)


class TestCommand(Command):
    description = "Run all package tests and exit immediately with informative return code."
    user_options = []
    
    def run(self):
        sys.exit(unitTests())
        
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    

class StyleCommand(Command):
    description = "Check all code for style, exit immediately with informative return code."
    user_options = []
    
    def run(self):
        sys.exit(checkStyle())
        
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass

    
class MergeTestCommand(Command):
    description = "Run all tests needed to determine whether the current code is suitable for merge."
    user_options = []
    
    def run(self):
        sys.exit(mergeTests())
        
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass

