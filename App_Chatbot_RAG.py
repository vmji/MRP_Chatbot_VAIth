import os #Zugriff auf Dateien für RAG
import sys
import torch
import accelerate #Für Anwendung der Berechnungen auf GPUs
from langchain_core.prompts import ChatPromptTemplate #Prompt Templates
from langchain_huggingface import HuggingFaceEmbeddings #Nutzung von Hugging Face Modellen

from langchain_core.runnables import chain
from langchain_community.chat_message_histories import ChatMessageHistory

#from langchain_community.document_loaders import GoogleDriveLoader, OneDriveFileLoader #nur falls es unbedingt notwendig, weil Zugriff auf Daten sehr schwierig ist

#Indizieren des Vektorraums mit Dokumenten
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

#Laden der eigenen Funktionen aus RAG_functions.py
import RAG_functions as rag

#Konfiguration der Modelle

#Import der Modelle aus der Datei RAG_functions.py
model_embedding = rag.model_embedding
model = rag.model

#Prompt Templates für Chats

# mit RAG

prompt_rag_chat = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your name is VAIth.
You are a helpful assistant and can use the following context to better assist the user:

<context>
{context}
</context>

If the context is irrelevant to the dialogue with the user, you may ignore it.
If the context is relevant to the interaction with the user you MUST refer to it.
The following is the current chat history:
{chat_history}
<|eot_id|><|start_header_id|>user<|end_header_id|>

User: {input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""")

# ohne RAG

prompt_chat = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your name is VAIth. Be concise in your answers. You are a helpful assistant and expected to assist the user.
The following is the current chat history:
{chat_history}
<|eot_id|><|start_header_id|>user<|end_header_id|>

User: {input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""")

# Frage ob RAG verwendet werden soll #Darauf folgt Neudefinition von CHAT_MODE

rag_user_answer = str(input("Do you want to reference your uploaded files and/or a created database in the Chat? (Y/n) "))
rag_answer = True if rag_user_answer.lower() in ["y", "yes", "ye", "ja"] else False
chat_mode = "RAG" if rag_answer==True else "Chat"

if rag_answer:
    num_k = 5 #int(input("How many documents do you wish to be referenced simultaneously: "))
else:
    num_k = 5

@chain
def retrieval_fuse_output(x: list, k = num_k): #Hilfsfunktion, welche bewirkt, dass nur die top num_k Dokumente im Prompt übernommen werden anstatt 2*nu_k von beiden retrievern
    """Only returns the top num_k Documents of EnsembleRetriever instead of num_k*2 of both retrievers"""
    return x[0:k]

# Falls ja einlesen der Dokumente bzw. sogar eines gesamten Vectorstores
# TO-DO: Je nachdem, ob beständige Vektordatenbank erstellt werden soll, eine Abfrage beim User machen. Abhängig von Antwort wird Variable für contextualization True oder False
# Bei beständiger Vektordatenbank kann ebenfalls Variable save_dir abgeleitet werden, welche entweder None oder default_path als Wert annimmt
# Zusätzlich prüfen, ob bereits eine Vektordatenbank vorliegt, falls JA muss rag_env_expand anstatt rag_env verwendet werden

FILE_PATH = sys.argv[1]
URL = sys.argv[2:]

# Hilfsgröße zur Angabe des Speicherpfads für personalisierte Vektordatenbank
vdb_save_dir = FILE_PATH.rsplit(sep='/', maxsplit=2)[0] #örtliches Userverzeichnis
username = FILE_PATH.split(sep='/')[3]
vdb_save_dir = vdb_save_dir + f"/{username}_vector_database" # Speicherverzeichnis der Vektordatenbank

if (os.listdir(FILE_PATH) == [] and len(URL) < 1): #Prüfen ob Ordner keine Dateien enthält UND keine URLs angegeben werden
    #Angaben der standardmäßigen Variablenwerte
    database_function = rag.rag_env #Speichern der zu verwendenden Funktion in einer Variable # Verwendung der Funktion zur Erstellung der Vektordatenbank, falls diese noch nicht existiert
    database_var = {'vector_store_type': FAISS, 'contextualization':False, 'save_dir':None} #Speichern der zu verwendenden Variablen für die Erweiterung der Vektordatenbank #mittels ** in Funktion einfügen
else: #Falls URLs oder Dokumente angegeben wurden, wird nach Erstellung/Erweiterung der Vektordatenbank nachgefragt
    vdb_user_answer = str(input("Do you want to create/expand a vector database with your uploaded files? (Y/n) "))
    vdb_contextualization = True if vdb_user_answer.lower() in ["y", "yes", "ye", "ja"] else False
    vdb_save_dir = vdb_save_dir if vdb_user_answer.lower() in ["y", "yes", "ye", "ja"] else None #Angabe des Speicherpfads der zu erstellenden Vektordatenbank
    #print(f"Vecotr Database will be stored in '{vdb_save_dir}'")

    #Fallunterscheidung ob neue Datenbank mittels rag_env erstellt oder mittels rag_env_expand erweitert werden soll
    if vdb_user_answer.lower() in ["y", "yes", "ye", "ja"]: # Fall, dass eine Datenbank erstellt/erweitert werden soll
        if os.path.exists(vdb_save_dir): # Prüfen, ob bereits eine Datenbank existiert
            database_function = rag.rag_env_expand #Speichern der zu verwendenden Funktion in einer Variable # Verwendung der Funktion zur Erweiterung der Vektordatenbank, falls diese bereits existiert
            database_var = {'vector_store_path': vdb_save_dir, 'vector_store_type': FAISS, 'contextualization':vdb_contextualization, 'save_dir':vdb_save_dir} #Speichern der zu verwendenden Variablen für die Erweiterung der Vektordatenbank
            # 'vector_store_path' wird für die Erweiterung der Vektordatenbank benötigt, weil dies den Pfad der zu erweiternden Vektordatenbank darstellt
            # Variablen 'directory' und 'file_paths' werden nicht definiert, weil diese essenzieller Teil der elif-Bedingungen sind
        else: #Fall, dass noch keine Vektordatenbank existiert
            database_function = rag.rag_env #Speichern der zu verwendenden Funktion in einer Variable # Verwendung der Funktion zur Erstellung der Vektordatenbank, falls diese noch nicht existiert
            database_var = {'vector_store_type': FAISS, 'contextualization':vdb_contextualization, 'save_dir':vdb_save_dir} #Speichern der zu verwendenden Variablen für die Erweiterung der Vektordatenbank #mittels ** in Funktion einfügen
            # Variablen 'directory' und 'file_paths' werden nicht definiert, weil diese essenzieller Teil der elif-Bedingungen sind
    else: #Fall, dass noch keine Vektordatenbank existiert
        database_function = rag.rag_env #Speichern der zu verwendenden Funktion in einer Variable # Verwendung der Funktion zur Erstellung der Vektordatenbank, falls diese noch nicht existiert
        database_var = {'vector_store_type': FAISS, 'contextualization':vdb_contextualization, 'save_dir':vdb_save_dir}

if (os.listdir(FILE_PATH) == [] and len(URL) < 1): #Prüfen ob Ordner keine Dateien enthält UND keine URLs angegeben werden
    #Fallunterscheidung zwischen personalisierter und allgemeiner Vektordatenbank
    if os.path.exists(vdb_save_dir):
        default_path = vdb_save_dir #Pfad zu dem personalisierten Vectorstore
    else:
        default_path = "/mount/point/veith/Chatbot_VAIth/Vector_Stores/Contextualized_Multilingual_Database" #Pfad zu einem bereits existierenden Vectorstore
    ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store_path=default_path,
                                   search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    print("Initialized default vector database.")

elif (os.listdir(FILE_PATH) != [] and len(URL) < 1): #Fall, dass Dateien angegeben werden, aber KEINE URLs
    vector_store, docs = database_function(directory=FILE_PATH, **database_var) #keine Kontextualisierung bei hochgeladenen Dateien zwecks Zeitersparnis
    ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store=vector_store, doc_splits=docs,
                                    embedding=model_embedding, 
                                    search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    print("Vectorized uploaded data.")

elif (os.listdir(FILE_PATH) == [] and not len(URL) < 1): #Fall, dass KEINE Dateien angegeben werden, aber URLs
    vector_store, docs = database_function(file_paths=URL, **database_var) #embedding = model_embedding falls ein spezifisches Modell verwendet werden soll #keine Kontextualisierung bei hochgeladenen Dateien zwecks Zeitersparnis
    ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store=vector_store, doc_splits=docs,
                                    embedding=model_embedding, 
                                    search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    print("Vectorized uploaded data.")

else: #Fall, dass sowohl Dateien UND URLs angegeben werden
    #Einzulesende Hilfsdokumente
    vector_store, docs = database_function(directory=FILE_PATH, file_paths=URL, **database_var) #embedding = model_embedding falls ein spezifisches Modell verwendet werden soll #keine Kontextualisierung bei hochgeladenen Dateien zwecks Zeitersparnis
    ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store=vector_store, doc_splits=docs,
                                    embedding=model_embedding, 
                                    search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    print("Vectorized uploaded data.")
retriever = ensemble_retriever | retrieval_fuse_output #neue Kette, welche bewirkt, dass nur die top num_k Dokumente im Prompt übernommen werden anstatt 2*nu_k von beiden retrievern

# Standard textgenerierungsparameter
streamer = rag.streamer
top_p = 0.9
top_k = 20

chat_history = ChatMessageHistory() #Initialisieren des Chatverlaufs
#Erstellen einer Hilfsfunktion, welche den Chatverlauf ausliest und an die RAG Chain übergibt
def get_chat_history(_):
    return chat_history


if __name__ == "__main__":

    # print("Before we begin, do you wish to further customize the text generation parameters?\n(Y/n)")
    # customization = str(input())
    # if customization.lower() in ["y", "yes", "ja", "ye"]: #anpassen der Textgenerierungsparameter
    #     temperature = float(input("Please enter your desired text generation temperature as a positive float value: ")) #Frage nach der gewünschten Temperatur
    #     max_tokens = int(input("Please enter the number your desired threshold of generated tokens: ")) #Frage nach der gewünschten menge an generierten Tokens
    #     print_sources_user = str(input("Do you want to see the document sources, which were used to generate the text? (Y/n) ")) #Frage ob die Dokumentenquellen mit angegeben werden sollen
    #     print_sources = True if print_sources_user.lower() in ["y", "yes", "ja", "ye"] else False #Transformation in True und False
    
    # else:
    #     temperature = 0.6
    #     max_tokens = 1000
    #     print_sources = True

    #Frage nach Textgenerierungsparametern wird ausgelassen
    temperature = 0.6
    max_tokens = 1000
    print_sources = True

    print("VAIth: Welcome to the Chat! How may I help you?")
    print("You can type 'exit' to end the conversation or 'switch' to switch the conversation mode to RAG-Chat or normal conversation")

    while True:    
        instruction = str(input("User: "))

        if instruction.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        elif instruction.lower() in ["switch", "sw", "change"]:
            chat_mode = "RAG" if chat_mode == "Chat" else "Chat" #Falls der Chatmodus initial "Chat" war wird dieser nun zu "RAG" gewechselt
            print(f"You are now in {chat_mode} conversation mode.")
            continue
        else:
            chat_history.add_user_message(instruction) #Hinzufügen der User-Eingabe in den Chatverlauf #Nur Aufnahme der Usereingabe wenn Sie keinen Befehl darstellt
        
        if chat_mode == "RAG": #Fallunterscheidung, welcher Prompt verwendet werden soll
            prompt = prompt_rag_chat #Prompt definiert als RAG prompt

            #Textgenerierung gemäß RAG
            print("VAIth: "); gen, response = rag.rag_gen(instruction, query_analysis=True, chat_history=get_chat_history,
                                        prompt=prompt, retriever=retriever, temperature=temperature, max_new_tokens=max_tokens, print_sources=print_sources,
                                        streamer=streamer)
            chat_history.add_ai_message(response) #Hinzufügen der Modellausgabe zum Chatverlauf
        else:
            prompt = prompt_chat
            #Normale Konversation
            llm = rag.model_pipe(do_sample=True,
                     streamer=streamer,
                     temperature = temperature,
                     top_p = top_p,
                     max_new_tokens = max_tokens,
                     top_k = top_k,
                     output_scores = False,
                     output_attention = False,
                     output_hidden_states = False,
                     )

            chat_chain = {"input": RunnablePassthrough(), "chat_history": get_chat_history} | prompt | llm
            print("VAIth: "); response = chat_chain.invoke(instruction)
            # response = llm.invoke(prompt.invoke({"input":instruction, "chat_history":chat_history}).to_string())
            chat_history.add_ai_message(response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")[-1]) #Hinzufügen der formatierten Modellausgabe zum Chatverlauf

            

# (Im Chat Modus) "switch" eingeben um von RAG zu normalem Chat zu wechseln oder umgekehrt
# input.lower() in ["switch", "sw", "change"]
# print(f"You are now in {current_mode}") mit Modi "RAG Chat mode" und "Chat mode"
#Falls Modus gewechselt wird muss aktiver prompt verändert werden

