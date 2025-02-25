@echo off 
setlocal enabledelayedexpansion

::Frage ob Installation ausgeführt werden soll
:ask
echo "Do you want to install the required files to use VAIth? (The installation may take up to multiple hours) (Y/n)"
set /p INSTALLATION=""
::Falls nein, Abbruch des Skripts
if /i "%INSTALLATION%"=="n" goto end
if /i not "%INSTALLATION%"=="Y" goto end

::Prüfen der Adminrechte
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :install
)

::Abfrage Skript als Admin auszuführen
echo Requesting administrative privileges...
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
"%temp%\getadmin.vbs"
del "%temp%\getadmin.vbs"
exit /b

:: Generierung des ssh keys falls noch nicht vorhanden

::Ausführen der Installationsskripts, falls diese noch nicht ausgeführt wurden call wird verwendet, um Programme auszuführen und dann weiterzuführen
:install
set "SCRIPT_DIR=%~dp0"
cd %SCRIPT_DIR%
call installation_script_desktop_link.cmd

:end