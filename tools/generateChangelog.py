import re, time, sys


def generateDebianChangelog(package, logFile, version, maintainer):
    """
    ------- Convert CHANGELOG format like:
    pyqtgraph-0.9.1  2012-12-29

    - change
    - change


    -------- to debian changelog format:
    python-pyqtgraph (0.9.1-1) UNRELEASED; urgency=low

    * Initial release.

    -- Luke <luke.campagnola@gmail.com>  Sat, 29 Dec 2012 01:07:23 -0500


    *package* is the name of the python package.
    *logFile* is the CHANGELOG file to read; must have the format described above.
    *version* will be used to check that the most recent log entry corresponds
              to the current package version.
    *maintainer* should be string like "Luke <luke.campagnola@gmail.com>".
    """
    releases = []
    current_version = None
    current_log = None
    current_date = None
    for line in open(logFile).readlines():
        match = re.match(package+r'-(\d+\.\d+\.\d+(\.\d+)?)\s*(\d+-\d+-\d+)\s*$', line)
        if match is None:
            if current_log is not None:
                current_log.append(line)
        else:
            if current_log is not None:
                releases.append((current_version, current_log, current_date))
            current_version, current_date = match.groups()[0], match.groups()[2]
            #sys.stderr.write("Found release %s\n" % current_version)
            current_log = []

    if releases[0][0] != version:
        raise Exception("Latest release in changelog (%s) does not match current release (%s)\n" % (releases[0][0],  version))

    output = []
    for release, changes, date in releases:
        date = time.strptime(date, '%Y-%m-%d')
        changeset = [
            "python-%s (%s-1) UNRELEASED; urgency=low\n" % (package, release),
            "\n"] + changes + [
            " -- %s  %s -0%d00\n"  % (maintainer, time.strftime('%a, %d %b %Y %H:%M:%S', date), time.timezone/3600),
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

        output.append(clean)
        output.append("")
    return "\n".join(output) + "\n"


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.stderr.write('Usage: generateChangelog.py package_name log_file version "Maintainer <maint@email.com>"\n')
        sys.exit(-1)

    print(generateDebianChangelog(*sys.argv[1:]))
