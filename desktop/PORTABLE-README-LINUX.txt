MultiSolid for Linux x86_64

Extract the entire tar.gz archive into a new folder, then launch ./MultiSolid
from inside the MultiSolid directory. Keep its files and subfolders together.
Python, DAETools, Graphviz, and the compiler used for Compiled runs are included.
The archive preserves executable permissions and symbolic links.

The Ubuntu 24.04 installer is the separate .deb download. Install it with:
  sudo apt install ./MultiSolid-<version>-linux-x64-setup.deb
Then open MultiSolid from the applications menu or run the multisolid command.
It installs under /opt/multisolid. Remove it with sudo apt remove multisolid.

Build on the oldest Linux distribution you intend to support. release.json
records minimum_glibc for this particular build; a newer operating system may
still have other differences, so consult the accompanying validation report.
The documented build baseline is Ubuntu 24.04 x86_64. A graphical desktop and
working graphics drivers are required. Headless acceptance testing uses Xvfb.

Create projects in a writable folder outside the application directory.
Projects contain inputs, results, plugins, and reusable compilation caches.
The first Compiled run may take several minutes. Later compatible runs reuse
the project's cache. Preferences use the usual per-user Qt/XDG settings.

To update a portable installation, extract a new application folder and open
your existing project. To remove it, delete its folder. Keep project folders
to retain your inputs and results. No root access is needed for portable use.
For the installer edition, close MultiSolid and install the newer .deb to update.

See LICENSE.txt, licenses/, compiled/licenses/, and graphviz/licenses/ for
licence texts. BUILD-INFO.json and release.json describe this build. Keep the
matching source and dependency build assets with redistributed binaries.

Source, build instructions and the GUI guide:
https://github.com/Griminir5/multisolid-CL
