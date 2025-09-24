@echo off
setlocal enabledelayedexpansion

:: Prüfen ob Desktopverbindung bereits existiert
if exist ".shortcut_created" (
    ::echo Shortcut already exists. Exiting.
    exit /b 0
)

:: Speichern des aktuellen Pfades in Variable
set "SCRIPT_DIR=%~dp0"

:: Erstellen desktop shortcut mit dynamischen paths
powershell -Command "$scriptDir = '%SCRIPT_DIR%'; $targetPath = Join-Path -Path $scriptDir -ChildPath 'App_Chatbot_RAG.bat'; $iconPath = Join-Path -Path $scriptDir -ChildPath 'App Icon.ico'; $WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('C:\Users\Public\Desktop\VAIth Chatbot.lnk'); $Shortcut.TargetPath = $targetPath; $Shortcut.IconLocation = $iconPath; $Shortcut.Save()"

:: Erstellen flag file um wiederholte Erstellung zu vermeiden
echo. > .shortcut_created