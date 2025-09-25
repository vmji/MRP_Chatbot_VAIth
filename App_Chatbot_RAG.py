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

Your name is VAIth. Be concise in your answers.
You are a helpful assistant and can use the following context to better assist the user:

<context>
{context}
</context>

If the context is irrelevant to the dialogue with the user, you may ignore it.
If the context is relevant to the interaction with the user you MUST refer to it and cite the passages you are referencing in your answer.
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

# Anpassen der Tokens an das verwendete Modell falls nicht default Llama verwendet wird
if "llama" not in rag.model_path.lower():
    prompts = [prompt_rag_chat, prompt_chat]

    # Ändern der Tokens
    for prompt in prompts:
        for key in rag.llama_translator:
            prompt.messages[0].prompt.template = prompt.messages[0].prompt.template.replace(key, rag.llama_translator[key])

# Frage ob RAG verwendet werden soll #Darauf folgt Neudefinition von CHAT_MODE

rag_user_answer = str(input("Do you want to reference your uploaded files and/or a created database in the Chat?\nThis will initiate the RAG conversation mode (Y/n) "))
rag_answer = True if rag_user_answer.lower() in ["y", "yes", "ye", "ja"] else False
chat_mode = "RAG" if rag_answer==True else "Chat"

if rag_answer:
    num_k = int(input("How many documents do you wish to be referenced simultaneously: "))
    num_k = min(num_k, 20) # maximal 20 Dokumente dürfen gleichzeitig referenziert werden
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

# Funktion zum Laden der default Vektordatenbank
def load_default_vector_store():
    """Loads the default vector store of the current user either if no external documents have been provided or if no new documents have been provided."""
    if os.path.exists(vdb_save_dir):
            default_path = vdb_save_dir #Pfad zu dem personalisierten Vectorstore
    else:
        default_path = "/mount/point/veith/Vector_Stores/Default_Chat_Database" # "/mount/point/veith/Chatbot_VAIth/Vector_Stores/Qwen3_Contextualized_Multilingual_Database" #Pfad zu einem bereits existierenden Vectorstore
    # Laden des default Vectorstores
    ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store_path=default_path,
                                search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    print("Initialized default vector database.")
    return ensemble_retriever, vector_store

if (os.listdir(FILE_PATH) == [] and len(URL) < 1): #Prüfen ob Ordner keine Dateien enthält UND keine URLs angegeben werden
    #Angaben der standardmäßigen Variablenwerte
    database_function = rag.rag_env #Speichern der zu verwendenden Funktion in einer Variable # Verwendung der Funktion zur Erstellung der Vektordatenbank, falls diese noch nicht existiert
    database_var = {'vector_store_type': FAISS, 'contextualization':False, 'save_dir':None} #Speichern der zu verwendenden Variablen für die Erweiterung der Vektordatenbank #mittels ** in Funktion einfügen
else: #Falls URLs oder Dokumente angegeben wurden, wird nach Erstellung/Erweiterung der Vektordatenbank nachgefragt
    vdb_user_answer = str(input("Do you want to create/expand a vector database with your uploaded files? (Y/n) "))
    vdb_contextualization = True if vdb_user_answer.lower() in ["y", "yes", "ye", "ja"] else False
    vdb_save_dir = vdb_save_dir if vdb_user_answer.lower() in ["y", "yes", "ye", "ja"] else None #Angabe des Speicherpfads der zu erstellenden Vektordatenbank
    # Wenn die Antwort nein lautet werden die hochgeladenen Dokumente zwar aufbereitet, aber nicht in die Vektordatenbank des Nutzers überführt, sondern diese bleiben isoliert in der aktuellen Sitzung
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
        default_path = "/mount/point/veith/Vector_Stores/Default_Chat_Database" #Pfad zu einem bereits existierenden Vectorstore
    ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store_path=default_path,
                                   search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    print("Initialized default vector database.")

elif (os.listdir(FILE_PATH) != [] and len(URL) < 1): #Fall, dass Dateien angegeben werden, aber KEINE URLs
    vector_store, docs = database_function(directory=FILE_PATH, **database_var) #keine Kontextualisierung bei hochgeladenen Dateien zwecks Zeitersparnis
    # Prüfen, ob neue Dokumente aufgenommen wurden
    if len(docs) > 0:
        ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store=vector_store, doc_splits=docs,
                                        embedding=model_embedding, 
                                        search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    else: # Fallback, falls keine neuen Dokumente eingelesen werden konnten
        ensemble_retriever, vector_store = load_default_vector_store()
    print("Vectorized uploaded data.")

elif (os.listdir(FILE_PATH) == [] and not len(URL) < 1): #Fall, dass KEINE Dateien angegeben werden, aber URLs
    vector_store, docs = database_function(file_paths=URL, **database_var) #embedding = model_embedding falls ein spezifisches Modell verwendet werden soll #keine Kontextualisierung bei hochgeladenen Dateien zwecks Zeitersparnis
    # Prüfen, ob neue Dokumente aufgenommen wurden
    if len(docs) > 0:
        ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store=vector_store, doc_splits=docs,
                                        embedding=model_embedding, 
                                        search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    else: # Fallback, falls keine neuen Dokumente eingelesen werden konnten
        ensemble_retriever, vector_store = load_default_vector_store()
    print("Vectorized uploaded data.")

else: #Fall, dass sowohl Dateien UND URLs angegeben werden
    #Einzulesende Hilfsdokumente
    vector_store, docs = database_function(directory=FILE_PATH, file_paths=URL, **database_var) #embedding = model_embedding falls ein spezifisches Modell verwendet werden soll #keine Kontextualisierung bei hochgeladenen Dateien zwecks Zeitersparnis
    # Prüfen, ob neue Dokumente aufgenommen wurden
    if len(docs) > 0:
        ensemble_retriever, vector_store = rag.load_retriever_vector_store(vector_store=vector_store, doc_splits=docs,
                                        embedding=model_embedding, 
                                        search_type="similarity", num_k=num_k, retrieval_scores=False) #Deaktivieren von Scores, weil diese inkompatibel mit EnsembleRetriever sind
    else: # Fallback, falls keine neuen Dokumente eingelesen werden konnten
        ensemble_retriever, vector_store = load_default_vector_store()
    print("Vectorized uploaded data.")
retriever = ensemble_retriever | retrieval_fuse_output #neue Kette, welche bewirkt, dass nur die top num_k Dokumente im Prompt übernommen werden anstatt 2*nu_k von beiden retrievern

# Standard textgenerierungsparameter
streamer = rag.streamer

chat_history = ChatMessageHistory() #Initialisieren des Chatverlaufs
#Erstellen einer Hilfsfunktion, welche den Chatverlauf ausliest und an die RAG Chain übergibt
def get_chat_history(_):
    return chat_history


if __name__ == "__main__":

    print("Before we begin, do you wish to further customize the text generation parameters?\n(Y/n)")
    customization = str(input())
    if customization.lower() in ["y", "yes", "ja", "ye"]: #anpassen der Textgenerierungsparameter
        if 'Qwen3'.lower() in rag.model_path.lower():
            detailed_thinking_question = str(input("Do you wish for the model to apply reasoning thinking during the chat? (Y/n) ")) # Frage ob Modell Denkprozesse während Textgenerierung ausüben soll
            detailed_thinking_qwen3 = "<|im_start|>assistant" if detailed_thinking_question.lower() in ["y", "yes", "ja", "ye"] else "<|im_start|>assistant\n<think>\n\n</think>" # Ersetzen eines anderen Promptelements, weil bei Qwen3 das Denken nicht im Systemprompt festgelegt wird
        temperature = float(input("Please enter your desired text generation temperature as a positive float value: ")) #Frage nach der gewünschten Temperatur
        max_tokens = int(input("Please enter the number your desired threshold of generated tokens: ")) #Frage nach der gewünschten menge an generierten Tokens
        print_sources = True
        # print_sources_user = str(input("Do you want to see the document sources, which were used to generate the text? (Y/n) ")) #Frage ob die Dokumentenquellen mit angegeben werden sollen
        # print_sources = True if print_sources_user.lower() in ["y", "yes", "ja", "ye"] else False #Transformation in True und False
    
    else:
        temperature = 0.6
        max_tokens = 1000
        print_sources = True
        # Defaultsmäßig werden Denkparameter für Generierung deaktiviert
        if 'Qwen3'.lower() in rag.model_path.lower():
            detailed_thinking_qwen3 = "<|im_start|>assistant\n<think>\n\n</think>" # Ersetzen eines anderen Promptelements, weil bei Qwen3 das Denken nicht im Systemprompt festgelegt wird # Standardmäßig wird das Denken deaktiviert
    
    # Anpassen der Textgenerierungsparameter gemäß Nutzerwünschen
    config_text_gen = {
        "temperature": temperature,
        "max_new_tokens": max_tokens,
    }
    llm = rag.llm_pipe_manager.get_pipeline(**config_text_gen) # Anpassen des Pipelinemanagers des Modells gemäß Nutzerwünschen

    llm_query = rag.llm_query_pipe_manager.get_pipeline()
    
    #Frage nach Textgenerierungsparametern wird ausgelassen
    # temperature = 0.6
    # max_tokens = 1000
    # print_sources = True

    # Verändern der Prompt Templates gemäß Wunsch nach detailliertem Denken
    if 'Qwen3'.lower() in rag.model_path.lower():
        prompt_rag_chat.messages[0].prompt.template = prompt_rag_chat.messages[0].prompt.template.replace('<|im_start|>assistant', detailed_thinking_qwen3) # einfügen der aktuellen Qwen3 Denkkonfiguration in den Prompt
        prompt_chat.messages[0].prompt.template = prompt_chat.messages[0].prompt.template.replace('<|im_start|>assistant', detailed_thinking_qwen3) # einfügen der aktuellen Qwen3 Denkkonfiguration in den Prompt

    print("VAIth: Welcome to the Chat! How may I help you?")
    print("You can type 'exit' to end the conversation. Type 'help' to see all available commands.")

    while True:    
        instruction = str(input("User: "))

        if instruction.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        elif instruction.lower() in ["switch", "sw", "change"]:
            chat_mode = "RAG" if chat_mode == "Chat" else "Chat" #Falls der Chatmodus initial "Chat" war wird dieser nun zu "RAG" gewechselt
            print(f"You are now in {chat_mode} conversation mode.")
            continue
        elif instruction.lower() in ["help", "info", "information"]:
            print("Below you can find all available commands, that you have access to during the conversation:")
            print("\nexit\t--Type 'exit' to end the conversation and close the application.")
            print("\nswitch\t--Type 'switch' to change the conversation mode between normal chat mode and RAG chat mode.")
            print("\nconfigure\t--Type 'configure' to change the text generation parameters.")
            # print("\n'save'\t--Type save to save the current chat history to a text file.")
            print("\n")
            continue

        elif instruction.lower() in ["configure", "config", "settings"]:
            print("Please enter your desired parameters for text generation:")
            # Textgenerierungspipeline Konfiguration
            if 'Qwen3'.lower() in rag.model_path.lower():
                old_detailed_thinking_qwen3 = detailed_thinking_qwen3
                detailed_thinking_question = str(input("Do you wish for the model to apply reasoning thinking during the chat? (Y/n) ")) # Frage ob Modell Denkprozesse während Textgenerierung ausüben soll
                detailed_thinking_qwen3 = "<|im_start|>assistant" if detailed_thinking_question.lower() in ["y", "yes", "ja", "ye"] else "<|im_start|>assistant\n<think>\n\n</think>" # Ersetzen eines anderen Promptelements, weil bei Qwen3 das Denken nicht im Systemprompt festgelegt wird
                if old_detailed_thinking_qwen3 != detailed_thinking_qwen3: # Fallunterscheidung, je nachdem ob detailliertes Denken initial bereits vorhanden ist oder nicht
                    # Verändern der Prompt Templates gemäß Wunsch nach detailliertem Denken
                    prompt_rag_chat.messages[0].prompt.template = prompt_rag_chat.messages[0].prompt.template.replace(old_detailed_thinking_qwen3, detailed_thinking_qwen3) # einfügen der aktuellen Qwen3 Denkkonfiguration in den Prompt
                    prompt_chat.messages[0].prompt.template = prompt_chat.messages[0].prompt.template.replace(old_detailed_thinking_qwen3, detailed_thinking_qwen3) # einfügen der aktuellen Qwen3 Denkkonfiguration in den Prompt

            temperature = float(input("Please enter your desired text generation temperature as a positive float value: ")) #Frage nach der gewünschten Temperatur
            max_tokens = int(input("Please enter the number your desired threshold of generated tokens: ")) #Frage nach der gewünschten menge an generierten Tokens

                # Anpassen der Textgenerierungsparameter gemäß Nutzerwünschen
            config_text_gen = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            }
            llm = rag.llm_pipe_manager.get_pipeline(**config_text_gen) # Anpassen des Pipelinemanagers des Modells gemäß Nutzerwünschen
            continue
        # TO-DO: Korrektheit des angegebenen Codes überprüfen.
        # Beachte: Export benötigt zusätzlichen ssh Befehl, um die Datei auf den lokalen Rechner zu übertragen.
        # elif instruction.lower() in ["save", "store", "export"]:
        #     save_path = str(input("Please enter the full path (including filename and .txt ending) where you want to save the chat history: "))
        #     try:
        #         with open(save_path, 'w') as f:
        #             for message in chat_history.messages:
        #                 role = "User" if message.type == "human" else "VAIth"
        #                 f.write(f"{role}: {message.content}\n")
        #         print(f"Chat history successfully saved to {save_path}")
        #     except Exception as e:
        #         print(f"An error occurred while saving the chat history: {e}")
        #     continue
        else:
            chat_history.add_user_message(instruction) #Hinzufügen der User-Eingabe in den Chatverlauf #Nur Aufnahme der Usereingabe wenn Sie keinen Befehl darstellt
        
        if chat_mode == "RAG": #Fallunterscheidung, welcher Prompt verwendet werden soll
            prompt = prompt_rag_chat #Prompt definiert als RAG prompt

            #Textgenerierung gemäß RAG
            print("VAIth: "); gen, response = rag.rag_gen(instruction, query_analysis=True, chat_history=get_chat_history,
                                        prompt=prompt, retriever=retriever, pipeline_textgen=llm, pipeline_rag_query=llm_query, print_sources=print_sources,)
            # Parsen vom Text falls das Modell Gedankengänge ausführt, damit das Kontextfenster nicht überladen wird
            if "Qwen3".lower() in rag.model_path.lower():
                response = response.split("</think>")[-1]
            chat_history.add_ai_message(response) #Hinzufügen der Modellausgabe zum Chatverlauf
        else:
            prompt = prompt_chat
            #Normale Konversation
            chat_chain = {"input": RunnablePassthrough(), "chat_history": get_chat_history} | prompt | llm
            if isinstance(instruction, str):
                print("VAIth: "); response = chat_chain.invoke(instruction)
            # response = llm.invoke(prompt.invoke({"input":instruction, "chat_history":chat_history}).to_string())
                chat_history.add_ai_message(response.split(f"{rag.parsing_string}\n")[-1]) #Hinzufügen der formatierten Modellausgabe zum Chatverlauf
            else: #theoretischer Natur falls in anderen Programmen Funktionalität einer kontextlosen Batchgenerierung notwendig wird
                print("VAIth: "); response = chat_chain.batch(instruction)

            

# (Im Chat Modus) "switch" eingeben um von RAG zu normalem Chat zu wechseln oder umgekehrt
# input.lower() in ["switch", "sw", "change"]
# print(f"You are now in {current_mode}") mit Modi "RAG Chat mode" und "Chat mode"
#Falls Modus gewechselt wird muss aktiver prompt verändert werden