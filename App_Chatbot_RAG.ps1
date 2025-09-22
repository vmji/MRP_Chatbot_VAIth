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

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
# Pause zum möglichen Debuggen
pause