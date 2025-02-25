Wilkommen bei VAIth dem MRP Chatbot!

Bevor wir damit beginnen die Nutzung des Chatbots zu erklären, sind folgend die Anforderungen aufgelistet, welche Sie erfüllen müssen:

Für externe Nutzung mit Fernverbindung:

!Sie brauchen einen Zugang zum DGX-Rechner des MRPs!
Dies bedeutet sie müssen ein registrierter User sein, um den Zugang zum ssh zu ermöglichen.
Außerdem benötigen Sie die Nutzerdaten, um in der Lage zu sein eigene Daten hochzuladen, zu welchen Sie dann VAIth ausfragen können.

Für lokale Nutzung:

Sie haben Python auf Ihrem Rechner installiert.
Sie haben sich vergewissert, dass CUDA_PATH, zumeist in "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x" gespeichert auf dem PATH Ihres Systems ist.
Sie haben git von "https://git-scm.com/downloads" installiert.
Sie haben git-lfs von "https://git-lfs.com/" installiert.
Damit alle notwendigen Dateien korrekt installiert werden können empfehlen wir das Programm bei der ersten Durchführung als Administrator auszuführen.

Glückwunsch Sie haben alle notwendigen Vorbereitungen abgeschlossen!


Folgend ist die vorhergesehene Nutzung von VAIth beschrieben.

Für externe Nutzung:

Zunächst werden Sie nach Ihrem Usernamen gefragt. Geben Sie hier den Usernamen an, mit welchem Sie Zugriff auf den DGX Rechner haben.
Sie werden nun nach Ihrem Passwort gefragt.

Für lokale und externe Nutzung:

Installieren Sie zunächst alle notwendigen Dateien und Modelle, indem Sie die Datei Installation_Wizard.cmd als Administrator ausführen.

Wenn Sie zum ersten mal dieses Programm starten wird ein Ordner in dem App-Verzeichnis erstellt.
Dieser Ordner befindet sich in "/[path]/Chatbot_VAIth/temporary_rag_files" und ist der Ordner, 
welcher Ihre hochgeladenen Dateien speichert.
Der Ordner speichert die Dateien jedoch nur für einzelne Sitzungen und wird zu Beginn jeder weiteren Sitzung geleert.

Sie werden nun gefragt ob Sie eigene Dateien hochladen möchten. Antworten Sie mit "Y" werden Sie nach Dateipfaden abgefragt.
Geben Sie bitte hierfür die Ordner- und oder Dateipfade an, in welchen sich Ihre Textdateien befinden. 
Achten Sie darauf, dass die Pfade nicht mit "/" enden.
Achten Sie außerdem darauf, dass in dem angegebenen Ordnerpfad nur Textdateien enthalten sind! 
Wenn andere Dateiformate wie bspw. .png enthalten sind kommt es zu einem Fehler in der Programmausführung.
Sie können so viele Ordnerpfade angeben wie sie möchten. Beachten Sie jedoch, dass Sie sich jedes mal mit dem Passwort einloggen müssen,
um die Ordnerinhalte hochzuladen. Sie können den Uploadprozess abbrechen indem Sie "exit" eingeben.
Sollten Sie die Frage, ob Sie Dateien hochladen möchten mit "n" beantworten, so arbeitet das Programm einfach weiter.

Sie werden noch ein letztes Mal nach Ihrem Passwort gefragt, bevor das eigentliche Programm startet.
Das Programm lädt zunächst die notwendigen Dateien.
Sie werden als nächstes gefragt, ob Sie ein spezifisches Embedding Modell verwenden möchten.
Die Wahl des Embedding Modells beeinflusst, wie die hochgeladenen Dateien eingelesen werden und wie gut die Dokumentensuche stattfinden wird.
Wenn Sie mit "Y" antworten werden Sie darum gebeten den Pfade des gewünschten Modells anzugeben.
Konsultieren Sie für mögliche Modellpfade die Datei "Embedding_Model_Directories.txt".
Wenn Sie auf die Frage nach dem Embedding Modell mit "n" antworten, wird das standardmäßige Embedding Modell geladen.

Danach werden Sie gefragt, ob Sie die hochgeladenen Dateien in Ihrem Chat mit VAIth referenzieren wollen.
Wenn Sie mit "Y" antworten, wird der Chat im RAG-Modus geladen und es werden die hochgeladenen Textdateien referenziert.
Wenn Sie mit "n" antworten, wird der Chat im normalen Chat-Modus geladen und das Modell greift auf die gelernten Informationen zurück.

Zuletzt folgt noch eine Frage, ob Sie die Textgenerierungsparameter anpassen wollen.
Antworten Sie mit "n" werden die Standardparameter geladen.
Antworten Sie mit "Y" können Sie die folgenden Parameter anpassen:
Temperatur, maximale Anzahl generierter Tokens, Quellenangabe im RAG-Modus.

Nun können Sie mit VAIth reden. Sie können den Konversationsmodus mittels dem Befehl "switch" wechseln.
Mit dem Befehl "exit" beenden Sie den Chat.

Ich hoffe VAIth kann Ihnen bei Ihrer Arbeit behilflich sein.