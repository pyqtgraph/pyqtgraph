import re, time, sys
if len(sys.argv) < 3:
    sys.stderr.write("Must specify changelog file and latest release!\n")
    sys.exit(-1)

### Convert CHANGELOG format like:
"""
pyqtgraph-0.9.1  2012-12-29

  - change
  - change
"""

### to debian changelog format:
"""
python-pyqtgraph (0.9.1-1) UNRELEASED; urgency=low

  * Initial release.

 -- Luke <luke.campagnola@gmail.com>  Sat, 29 Dec 2012 01:07:23 -0500
"""



releases = []
current_version = None
current_log = None
current_date = None
for line in open(sys.argv[1]).readlines():
    match = re.match(r'pyqtgraph-(\d+\.\d+\.\d+(\.\d+)?)\s*(\d+-\d+-\d+)\s*$', line)
    if match is None:
        if current_log is not None:
            current_log.append(line)
    else:
        if current_log is not None:
            releases.append((current_version, current_log, current_date))
        current_version, current_date = match.groups()[0], match.groups()[2]
        #sys.stderr.write("Found release %s\n" % current_version)
        current_log = []

if releases[0][0] != sys.argv[2]:
    sys.stderr.write("Latest release in changelog (%s) does not match current release (%s)\n" % (releases[0][0],  sys.argv[2]))
    sys.exit(-1)

for release, changes, date in releases:
    date = time.strptime(date, '%Y-%m-%d')
    changeset = [ 
        "python-pyqtgraph (%s-1) UNRELEASED; urgency=low\n" % release,
        "\n"] + changes + [
        " -- Luke <luke.campagnola@gmail.com>  %s -0%d00\n"  % (time.strftime('%a, %d %b %Y %H:%M:%S', date), time.timezone/3600),
        "\n" ]

    # remove consecutive blank lines except between releases
    clean = ""
    lastBlank = True
    for line in changeset:
        if line.strip() == '':
            if lastBlank:
                continue
            else:
                clean += line
            lastBlank = True
        else:
            clean += line
            lastBlank = False
            
    print clean 
    print ""


