# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus and Kyle Kastner
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$MESA_GL_URL = "https://github.com/vispy/demo-data/raw/master/mesa/"

# Mesa DLLs found linked from:
#     http://qt-project.org/wiki/Cross-compiling-Mesa-for-Windows
# to:
#     http://sourceforge.net/projects/msys2/files/REPOS/MINGW/x86_64/mingw-w64-x86_64-mesa-10.2.4-1-any.pkg.tar.xz/download

function DownloadMesaOpenGL ($architecture) {
    [Net.ServicePointManager]::SecurityProtocol = 'Ssl3, Tls, Tls11, Tls12'
    $webclient = New-Object System.Net.WebClient
    $basedir = $pwd.Path + "\"
    $filepath = $basedir + "opengl32.dll"
    # Download and retry up to 3 times in case of network transient errors.
    $url = $MESA_GL_URL + "opengl32_mingw_" + $architecture + ".dll"
    Write-Host "Downloading" $url
    $retry_attempts = 2
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } else {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
}


function main () {
    DownloadMesaOpenGL $env:PYTHON_ARCH
}

main
