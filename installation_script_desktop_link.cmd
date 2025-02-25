@echo off
setlocal enabledelayedexpansion

:: Prüfen ob Desktopverbindung bereits existiert
if exist ".shortcut_created" (
    ::echo Shortcut already exists. Exiting.
    exit /b 0
)

:: Speichern des aktuellen Pfades in Variable
set "SCRIPT_DIR=%~dp0"

:: Speichern des Parent Pfades in einer Variable
for %%I in ("%SCRIPT_DIR%..") do set "PARENT_DIR=%%~fI\"

:: Speichern des Zielpfads in Variable
set "TARGET_PATH=%PARENT_DIR%App_Chatbot_RAG.cmd"

:: Speichern des Iconpfads in Variable
set "ICON_PATH=%PARENT_DIR%App Icon.ico"


:: Erstellen desktop shortcut mit dynamischen paths
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('C:\Users\Public\Desktop\VAIth Chatbot.lnk'); $Shortcut.TargetPath = '!TARGET_PATH!'; $Shortcut.IconLocation = '!ICON_PATH!'; $Shortcut.Save()"
::$WshShell.CreateShortcut('%UserProfile%\Desktop\VAIth Chatbot.lnk')

:: Erstellen flag file um wiederholte Erstellung zu vermeiden
echo. > .shortcut_created