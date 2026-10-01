#ifndef BundleDir
  #error Supply /DBundleDir with the verified application folder
#endif
#ifndef AppVersion
  #error Supply /DAppVersion
#endif
#ifndef BuildId
  #error Supply /DBuildId
#endif
#define VersionFolder AppVersion + "-" + BuildId

[Setup]
AppId={{E09BF0B1-78BE-4A87-960F-542D9563271B}
AppName=MultiSolid
AppVersion={#AppVersion}
AppPublisher=MultiSolid contributors
DefaultDirName={localappdata}\Programs\MultiSolid
DefaultGroupName=MultiSolid
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0.22000
DisableProgramGroupPage=yes
LicenseFile={#BundleDir}\LICENSE.txt
OutputBaseFilename=MultiSolid-{#AppVersion}-windows-x64-setup
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\app\{#VersionFolder}\MultiSolid.exe
CloseApplications=yes
RestartApplications=no
VersionInfoVersion={#AppVersion}.0

[Files]
; Versioned application directories keep upgrades from mixing old and new DLLs.
; Inno's uninstall log removes only installed files; user projects are never targeted.
Source: "{#BundleDir}\*"; DestDir: "{app}\app\{#VersionFolder}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Icons]
Name: "{autoprograms}\MultiSolid"; Filename: "{app}\app\{#VersionFolder}\MultiSolid.exe"; WorkingDir: "{app}\app\{#VersionFolder}"
Name: "{autodesktop}\MultiSolid"; Filename: "{app}\app\{#VersionFolder}\MultiSolid.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\app\{#VersionFolder}\MultiSolid.exe"; Description: "Launch MultiSolid"; Flags: nowait postinstall skipifsilent
