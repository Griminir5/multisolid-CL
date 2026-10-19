"""Generate Windows executable resources from the application's version and SVG."""
from importlib import metadata

from packaging.version import Version


def numeric_version(version):
    """Windows fixed resources require four 16-bit integers, without rc/dev labels."""
    components = Version(version).release
    if len(components) > 4 or any(value > 65535 for value in components):
        raise ValueError(f"Version does not fit the Windows numeric resource: {version}")
    return components + (0,) * (4 - len(components))


def generate(root):
    from PIL import Image
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer
    image = QImage(256, 256, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    QSvgRenderer(str(root / "desktop/packed_bed_ui/assets/multisolid.svg")).render(painter)
    painter.end()
    png = root / "build/multisolid.png"
    png.parent.mkdir(exist_ok=True)
    image.save(str(png))
    icon = png.with_suffix(".ico")
    Image.open(png).save(icon, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    version = metadata.version("multisolid-cl-ui")
    components = numeric_version(version)
    flags = 0x2 if Version(version).is_prerelease else 0x0
    version_file = root / "build/windows-version.txt"
    version_file.write_text(f'''VSVersionInfo(
      ffi=FixedFileInfo(filevers={components!r}, prodvers={components!r}, mask=0x3f,
                       flags={flags:#x}, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
      kids=[StringFileInfo([StringTable('040904B0', [
        StringStruct('CompanyName', 'MultiSolid contributors'),
        StringStruct('FileDescription', 'MultiSolid packed-bed simulation'),
        StringStruct('FileVersion', {version!r}),
        StringStruct('InternalName', 'MultiSolid'),
        StringStruct('OriginalFilename', 'MultiSolid.exe'),
        StringStruct('ProductName', 'MultiSolid'),
        StringStruct('ProductVersion', {version!r})])]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])])])
''', encoding="utf-8")
    return icon, version_file
