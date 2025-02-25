@echo off 
setlocal enabledelayedexpansion 

:: Verwenden des korrekten Char Encodings
chcp 65001
set "FILE_PATHS="
set /p USERNAME= "Enter your username: " 
echo '%USERNAME%'

set REMOTE_DIR=/mount/point/%USERNAME%/temporary_rag_files
echo Uploaded files will be stored in '%REMOTE_DIR%'

call ssh %USERNAME%@10.246.58.13 "mkdir -p %REMOTE_DIR%; cd %REMOTE_DIR%/; rm -rf *; exit"

:ask
set /p UPLOAD_FILES= "Do you want to upload datafiles from your local machine? (Y/n) "
if /i "%UPLOAD_FILES%"=="n" goto end
if /i not "%UPLOAD_FILES%"=="Y" goto end

echo Reminder to have your files stored inside a folder. All the files need to be in a readable text format like .txt, .pdf, .docx

:loop
set /p FILE_PATH_FROM="Enter the path to the file or directory where your data is stored (type 'exit' to finish): "
set "FILE_PATH_FROM=%FILE_PATH_FROM:"=%"
if /i "%FILE_PATH_FROM%"=="exit" goto end
::set "FILE_PATHS=!FILE_PATHS! %FILE_PATH_FROM%"
echo "Copying data from: !FILE_PATH_FROM!\ to:%REMOTE_DIR%/"
::Befehl kopiert kompletten Ordner
scp -r "!FILE_PATH_FROM!\*" %USERNAME%@10.246.58.13:%REMOTE_DIR%/
goto loop
:end

if defined FILE_PATHS (
    echo Upload completed.
) else (
    echo Continuing without uploading data...
)

:: Abfrage ob URLs als Datenquellen verwendet werden sollen
:ask_url
::Definition der Listengröße für den Loop
set "urls="
set /p PASS_URLs= "Do you want to pass URLs to the application? (Y/n) "
if /i "%PASS_URLs%"=="n" goto end_url
if /i not "%PASS_URLs%"=="Y" goto end_url

:loop_url
set /p URL="Enter the URL for the website you wish to be referenced (type 'exit' to finish): "
set "URL=%URL:"=%"
if /i "%URL%"=="exit" goto end_url
set "urls=!urls! !URL!"
goto loop_url
:end_url

ssh %USERNAME%@10.246.58.13 "/mount/point/veith/.venv/bin/python /mount/point/veith/MRP_Chatbot_VAIth/App_Chatbot_RAG.py %REMOTE_DIR%/ %urls%"
pause