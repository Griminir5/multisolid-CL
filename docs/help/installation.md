# Install the desktop application

The first public desktop release is pending. Once published, download it from
[GitHub Releases](https://github.com/Griminir5/multisolid-CL/releases).

## Windows

The Windows distribution targets **Windows 11 x64**. Choose one release asset:

| Asset | Installation |
| --- | --- |
| `MultiSolid-<version>-windows-x64-setup.exe` | Run the installer, then open MultiSolid from the Start menu. Installation is for the current user. |
| `MultiSolid-<version>-windows-x64-portable.zip` | Extract the entire ZIP into a new folder, then run `MultiSolid/MultiSolid.exe`. Keep its files and subfolders together. |

## Linux

The Linux distribution targets **Ubuntu 24.04 x86_64**. Choose one release asset:

| Asset | Installation |
| --- | --- |
| `MultiSolid-<version>-linux-x64-setup.deb` | Run `sudo apt install ./MultiSolid-<version>-linux-x64-setup.deb`, then open MultiSolid from the applications menu. |
| `MultiSolid-<version>-linux-x64-portable.tar.gz` | Extract the entire archive, then run `./MultiSolid/MultiSolid`. Keep its files and subfolders together. |

The installer adds the `multisolid` command and installs under `/opt/multisolid`.
Remove it with `sudo apt remove multisolid`. Other Linux distributions are not
qualified by the Ubuntu build.

## Projects and updates

The desktop distributions include Python, DAETools, Graphviz, and the compiler
and libraries needed for Compiled execution. You do not need to install these
separately. GitHub's automatically generated source archives contain source code;
choose an installer or portable archive for the ready-to-run application.

Create projects in a writable folder outside the application directory. The first
Compiled run may take several minutes to build its model; compatible later runs
reuse the project's cache. To update a portable installation, extract the new
version into a new folder and open your existing project.

Follow [Getting started](getting-started.md) to create and run your first case,
or use the [GUI guide](../GUI-GUIDE.md) for the full project workflow.
