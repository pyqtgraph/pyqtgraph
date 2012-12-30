from  subprocess import check_output
import re, time

def run(cmd):
    return check_output(cmd, shell=True)

tags = run('bzr tags')
versions = []
for tag in tags.split('\n'):
    if tag.strip() == '':
        continue
    ver, rev = re.split(r'\s+', tag)
    if ver.startswith('pyqtgraph-'):
        versions.append(ver)

for i in range(len(versions)-1)[::-1]:
    log = run('bzr log -r tag:%s..tag:%s' % (versions[i], versions[i+1]))
    changes = []
    times = []
    inmsg = False
    for line in log.split('\n'):
        if line.startswith('message:'):
            inmsg = True
            continue
        elif line.startswith('-----------------------'):
            inmsg = False
            continue
        
        if inmsg:
            changes.append(line)
        else:
            m = re.match(r'timestamp:\s+(.*)$', line)
            if m is not None:
                times.append(m.groups()[0])

    citime = time.strptime(times[0][:-6], '%a %Y-%m-%d %H:%M:%S')

    print "python-pyqtgraph (%s-1) UNRELEASED; urgency=low" % versions[i+1].split('-')[1]
    print ""
    for line in changes:
        for n in range(len(line)):
            if line[n] != ' ':
                n += 1
                break

        words = line.split(' ')
        nextline = ''
        for w in words:
            if len(w) + len(nextline) > 79:
                print nextline
                nextline = (' '*n) + w
            else:
                nextline += ' ' + w
        print nextline
    #print '\n'.join(changes)
    print ""
    print " -- Luke <luke.campagnola@gmail.com>  %s -0%d00" % (time.strftime('%a, %d %b %Y %H:%M:%S', citime), time.timezone/3600) 
    #print " -- Luke <luke.campagnola@gmail.com>  %s -0%d00" % (times[0], time.timezone/3600) 
    print ""

print """python-pyqtgraph (0.9.0-1) UNRELEASED; urgency=low

  * Initial release.

 -- Luke <luke.campagnola@gmail.com>  Thu, 27 Dec 2012 02:46:26 -0500"""

