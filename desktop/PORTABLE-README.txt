MultiSolid for Windows x64

For the portable ZIP, extract it fully, then double-click MultiSolid.exe. Keep
all files and subfolders together. For the installed edition, launch MultiSolid
from the Start menu. No separate Python, Graphviz or compiler installation is
required, and normal use does not require administrator privileges or internet.

Create a project in a writable folder outside this application folder. Projects
contain inputs, results, plugins and compilation caches. Preferences and recent
projects use your Windows user settings. The portable edition does not register
an uninstaller or modify PATH; the installer registers a per-user uninstaller.

To update the portable edition, extract the new version into a new folder and
open your existing project. To remove it, delete the application folder. Keep
your project folders if you want to retain inputs and results. For the installed
edition, use the new installer to update and Windows Settings to uninstall.

The first Compiled run may take several minutes while the included compiler
builds the model. Later compatible runs reuse the project cache.

These initial builds are unsigned. See release.json for the exact version and
build ID, LICENSE.txt and licenses for notices, and the accompanying validation
report for what was tested. Source code and build instructions are maintained
separately from this application folder.
