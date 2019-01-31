import os, sys
import subprocess as sp

_SHELL_CMD = '/bin/bash'
if os.name == 'nt':
    _SHELL_CMD = 'cmd'


def shell(cmd):
    """Run each line of a shell script; raise an exception if any line returns
    a nonzero value.
    """
    pin, pout = os.pipe()
    proc = sp.Popen(_SHELL_CMD, stdin=sp.PIPE)
    for line in cmd.split('\n'):
        line = line.strip()
        if line.startswith('#'):
            print('\033[33m> ' + line + '\033[0m')
        else:
            print('\033[32m> ' + line + '\033[0m')
        if line.startswith('cd '):
            os.chdir(line[3:])
        proc.stdin.write((line + '\n').encode('utf-8'))
        proc.stdin.write(('echo $? 1>&%d\n' % pout).encode('utf-8'))
        ret = ""
        while not ret.endswith('\n'):
            ret += os.read(pin, 1)
        ret = int(ret.strip())
        if ret != 0:
            print("\033[31mLast command returned %d; bailing out.\033[0m" % ret)
            sys.exit(-1)
    proc.stdin.close()
    proc.wait()


def ssh(host, cmd):
    """Run commands on a remote host by ssh.
    """
    proc = sp.Popen(['ssh', host], stdin=sp.PIPE)
    proc.stdin.write(cmd)
    proc.wait()

