# UTF-8 encoding implementieren
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Laden Variablen aus .env Datei
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and !$line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Length -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim()
                Set-Item -Path "env:$key" -Value $value
            }
        }
    }
}
$MRP_IP = $env:MRP_IP
$USERNAME = $env:USERNAME

# Initialisieren der Variablen
$filePaths = @()
#$USERNAME = Read-Host -Prompt "Enter your username"
Write-Host "Welcome '$USERNAME'!"

$downloadDir = [System.IO.Path]::Combine($env:USERPROFILE, "Downloads")
$REMOTE_DIR = "/mount/point/$USERNAME/temporary_rag_files"
Write-Host "Uploaded files will be stored in '$REMOTE_DIR'"

# Erstellen und saeubern des User-Verzeichnisses
ssh "$USERNAME@$MRP_IP" "mkdir -p $REMOTE_DIR; cd $REMOTE_DIR/; rm -rf *.docx *.pdf *.txt *.json *.csv; exit"

# Frage nach Dateiuploads
$UPLOAD_FILES = Read-Host -Prompt "Do you want to upload datafiles from your local machine? (Y/n)"
if ($UPLOAD_FILES -eq "Y") {
    Write-Host "Reminder: Files can be individual files or directories. All files need to be in a readable text format like .txt, .pdf, .docx"
    
    # Loop for file uploads
    while ($true) {
        $FILE_PATH_FROM = Read-Host -Prompt "Enter the path to the file or directory where your data is stored. You can also drag and drop the files (type 'exit' to finish)"
        $FILE_PATH_FROM = $FILE_PATH_FROM -replace '"', ''
        
        if ($FILE_PATH_FROM -eq "exit") {
            break
        }
        
        # Check if path exists
        if (-not (Test-Path $FILE_PATH_FROM)) {
            Write-Host "Error: Path '$FILE_PATH_FROM' does not exist. Please try again." -ForegroundColor Red
            continue
        }
        
        # Check if it's a file or directory and handle accordingly
        if (Test-Path $FILE_PATH_FROM -PathType Leaf) {
            # It's a file
            Write-Host "Copying file: $FILE_PATH_FROM to: $REMOTE_DIR/"
            scp "$FILE_PATH_FROM" "$USERNAME@${MRP_IP}:$REMOTE_DIR/"
        } elseif (Test-Path $FILE_PATH_FROM -PathType Container) {
            # It's a directory
            Write-Host "Copying directory contents from: $FILE_PATH_FROM\ to: $REMOTE_DIR/"
            scp -r "$FILE_PATH_FROM\*" "$USERNAME@${MRP_IP}:$REMOTE_DIR/"
        } else {
            Write-Host "Error: Unable to determine if '$FILE_PATH_FROM' is a file or directory." -ForegroundColor Red
        }
    }
    
    Write-Host "Upload completed."
} else {
    Write-Host "Continuing without uploading data..."
}

# URL-Abfrage
$urls = @()
$PASS_URLs = Read-Host -Prompt "Do you want to pass URLs to the application? (Y/n)"
if ($PASS_URLs -eq "Y") {
    # Schleifendurchlauf zum Sammeln der URLs
    while ($true) {
        $URL = Read-Host -Prompt "Enter the URL for the website you wish to be referenced (type 'exit' to finish)"
        $URL = $URL -replace '"', ''
        
        if ($URL -eq "exit") {
            break
        }
        
        $urls += $URL
    }
}

# Verbinden der URLs
$urlsString = $urls -join ' '

ssh $USERNAME@$MRP_IP "/mount/point/veith/.venv/bin/python /mount/point/veith/MRP_Chatbot_VAIth/App_Chatbot_RAG.py $REMOTE_DIR/ $urls"

# Prüfen, ob Chatverläufe gespeichert wurden durch auslesen der Manifest Datei
$REMOTE_MANIFEST_PATH = "/mount/point/$USERNAME/exports.manifest"
$LOCAL_ARCHIVE_NAME = "$downloadDir\chat_exports_$(Get-Date -Format 'yyyyMMddHHmmss').tar.gz"

Write-Host "Checking for exported chat files to download..."
# Definieren vom multi-line command, welches auf dem externen Gerät ausgeführt werden soll
$REMOTE_COMMAND = @"
if [ ! -f '$REMOTE_MANIFEST_PATH' ]; then
    exit 1;
fi;

cd '$REMOTE_DIR/..';
tar -czf temp_archive.tar.gz -T exports.manifest 2>/dev/null;
cat temp_archive.tar.gz;
rm temp_archive.tar.gz exports.manifest;
"@

$exitCode = 0
try {
    # Ausführen des remote commands und speichern der Ausgabe in einer lokalen Datei
    $process = Start-Process -FilePath "ssh" -ArgumentList "-q", "$USERNAME@10.246.57.247", $REMOTE_COMMAND -RedirectStandardOutput $LOCAL_ARCHIVE_NAME -RedirectStandardError "nul" -Wait -PassThru -WindowStyle Hidden
    $exitCode = $process.ExitCode
} catch {
    $exitCode = 1
}

if ($exitCode -eq 1) {
    Write-Host "No manifest file found. Nothing to download."
    # Entfernen der lokalen Datei, falls sie leer oder unvollständig ist
    if (Test-Path $LOCAL_ARCHIVE_NAME) {
        Remove-Item $LOCAL_ARCHIVE_NAME
    }
} else {
    Write-Host "Manifest file found. Files archived and downloaded."
    # Extrahieren des tar.gz-Archivs
    Write-Host "Extracting files..."
    try {
        # Auflisten der Dateien im Archiv
        $listProcess = Start-Process -FilePath "tar" -ArgumentList "-tzf", $LOCAL_ARCHIVE_NAME -RedirectStandardOutput "temp_file_list.txt" -Wait -PassThru -NoNewWindow
        
        if ($listProcess.ExitCode -eq 0 -and (Test-Path "temp_file_list.txt")) {
            # Auslesen der Dateiliste und Filtern nach .txt Dateien
            $fileList = Get-Content "temp_file_list.txt"
            $txtFiles = $fileList | Where-Object { $_ -match '\.txt$' }
            
            if ($txtFiles.Count -gt 0) {
                # Zunächst ein temporäres Verzeichnis erstellen im Download Ordner des aktuellen Users
                $tempDir = "$downloadDir\temp_extracted_chat_files_$(Get-Date -Format 'yyyyMMddHHmmss')"
                New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

                # Alle Dateien in das temporäre Verzeichnis extrahieren
                $extractProcess = Start-Process -FilePath "tar" -ArgumentList "-xzf", $LOCAL_ARCHIVE_NAME, "-C", $tempDir -Wait -PassThru -NoNewWindow
                
                if ($extractProcess.ExitCode -eq 0) {
                    # extrahierte .txt Dateien in das aktuelle Verzeichnis verschieben
                    $extractedTxtFiles = Get-ChildItem -Path $tempDir -Recurse -Filter "*.txt"
                    
                    foreach ($file in $extractedTxtFiles) {
                        $targetPath = Join-Path (Get-Location) $file.Name
                        Move-Item -Path $file.FullName -Destination $targetPath -Force
                        Write-Host "Extracted: $($file.Name)" -ForegroundColor Green
                    }
                    
                    # Löschen des temporären Verzeichnisses
                    Remove-Item -Path $tempDir -Recurse -Force
                    
                    Write-Host "Chat files have been extracted to the current directory." -ForegroundColor Green
                    # Löschen der lokalen Archivdatei
                    Remove-Item $LOCAL_ARCHIVE_NAME
                } else {
                    Write-Host "Error extracting archive. Archive file retained." -ForegroundColor Red
                }
            } else {
                Write-Host "No .txt files found in the archive." -ForegroundColor Yellow
            }
            
            # Löschen der temporären Datei mit der Dateiliste
            Remove-Item "temp_file_list.txt" -ErrorAction SilentlyContinue
        } else {
            Write-Host "Error listing archive contents. Archive file retained." -ForegroundColor Red
        }
    } catch {
        Write-Host "Error: Unable to extract archive. Please extract manually using: tar -xzf $LOCAL_ARCHIVE_NAME" -ForegroundColor Red
    }
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
# Pause zum möglichen Debuggen
pause