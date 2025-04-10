import os #Zugriff auf Dateien für RAG
os.environ['USER_AGENT'] = 'myagent'

import torch
import gc #GPU Memory Optimierung
import pandas
from tqdm.auto import tqdm #Fortschrittsbalken für Promptausgaben

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline, BitsAndBytesConfig #Funktionalität Hugging Face Modelle
# import accelerate #Für Anwendung der Berechnungen auf GPUs
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate #Prompt Templates
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings #Nutzung von Hugging Face Modellen

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader, TextLoader, Docx2txtLoader #Laden von PDF Dateien, Webseiten, Pandas Datensätzen, .txt Dateien, Word Dokumente
from langchain_community.document_loaders.csv_loader import CSVLoader # Laden von .csv Dateien
from langchain_community.document_loaders.json_loader import JSONLoader # Laden von .json Dateien
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_message_histories import ChatMessageHistory #Chast History

#from langchain_community.document_loaders import GoogleDriveLoader, OneDriveFileLoader #nur falls es unbedingt notwendig, weil Zugriff auf Daten sehr schwierig ist

#Indizieren des Vektorraums mit Dokumenten
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Funktionen Quellenangaben
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, chain, RunnablePick

#Konfiguration der Modelle

print("Initializing embedding model...")
model_kwargs = {'device': 'cuda:3'} #Argumente für Embedding Modell
encode_kwargs = {'normalize_embeddings': True} #Argumente für Codierung der Textsequenzen
model_embedding = HuggingFaceEmbeddings(model_name="/mount/point/veith/Models/multilingual-e5-large-instruct", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs) #auslagern der Rechenleistung auf GPU
#wird benutzt falls eigene Dateien hochgeladen wurden, aber das standardmäßige Embeddingmodell verwendet wird
tokenizer_embedding = AutoTokenizer.from_pretrained("/mount/point/veith/Models/multilingual-e5-large-instruct")

# print("Do you want to use a specific Embedding Model to vectorize your document(s) (Y/n)?")
# answ_embedding = str(input())

# if answ_embedding.lower() in ["y", "yes", "ye", "ja"]: #Fallunterscheidung ob default Embedding verwendet werden soll

#     print("Please enter the path to your preferred embedding model: Type 'exit' if you want to use the default embedding model.")
#     answ_model_embedding_path = "wrong/path" #default falsche Antwort

#     while os.path.exists(answ_model_embedding_path) == False: #Schleife um so lange nach validem Modellpfad zu fragen bis einer angegeben wurde
#         answ_model_embedding_path = str(input())

#         if os.path.exists(answ_model_embedding_path):#"Please enter a valid path" #sicherstellen, dass der Pfad existiert
#             print("initializing embedding model...")
#             model_kwargs = {'device': 'cuda:0', # 'cuda:3'
#                             "trust_remote_code" : True,} #Argumente für Embedding Modell
#             encode_kwargs = {'normalize_embeddings': True} #Argumente für Codierung der Textsequenzen

#             model_embedding = HuggingFaceEmbeddings(model_name=answ_model_embedding_path,
#                 model_kwargs=model_kwargs,
#                 encode_kwargs=encode_kwargs,
#                 # multi_process = True,
#                 show_progress = True,
#                 )
#             tokenizer_embedding = AutoTokenizer.from_pretrained(answ_model_embedding_path)
            
#         elif answ_model_embedding_path.lower() in ['exit', 'ex', 'e']: #Abbruchkriterium für User
#             print("Initializing default model...")
#             model_kwargs = {'device': 'cuda:0'} #Argumente für Embedding Modell
#             encode_kwargs = {'normalize_embeddings': True} #Argumente für Codierung der Textsequenzen
#             model_embedding = HuggingFaceEmbeddings(model_name="/mount/point/veith/Chatbot_VAIth/multilingual-e5-large-instruct", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs) #auslagern der Rechenleistung auf GPU
#             tokenizer_embedding = AutoTokenizer.from_pretrained("/mount/point/veith/Chatbot_VAIth/multilingual-e5-large-instruct") #kleines CPU freundliches Embedding Modell
#             break

#         else:
#             print("Entered invalid path. Please enter a valid path. Type 'exit' if you want to use the default embedding model.")
# else: #default Embedding Modell
#     print("Initializing default model...")
#     model_kwargs = {'device': 'cuda:0'} #Argumente für Embedding Modell
#     encode_kwargs = {'normalize_embeddings': True} #Argumente für Codierung der Textsequenzen
#     model_embedding = HuggingFaceEmbeddings(model_name="/mount/point/veith/Chatbot_VAIth/multilingual-e5-large-instruct", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs) #auslagern der Rechenleistung auf GPU
#     #wird benutzt falls eigene Dateien hochgeladen wurden, aber das standardmäßige Embeddingmodell verwendet wird
#     tokenizer_embedding = AutoTokenizer.from_pretrained("/mount/point/veith/Chatbot_VAIth/multilingual-e5-large-instruct")

#Initialisierung des Modells

print("Initializing LLM...")
# Frage nach zu verwendendem Modell
model_question = int(input("Which text generation model do you want to use?\n1.\tLlama-3.1-Nemotron-Nano-8B-v1\n2.\tLlama-3.1-8B-Instruct\n3.\tLlama-3.1-Nemotron-70B-Instruct-HF (4bit quantized)\nPlease answer by typing in the corresponding model number: "))

if model_question == 1:
    model_path = "/mount/point/veith/Models/Llama-3.1-Nemotron-Nano-8B-v1"

elif model_question == 2:
    model_path = '/mount/point/veith/Models/Llama-3.1-8B-Instruct'

elif model_question == 3:
    model_path = "/mount/point/veith/Models/Llama-3.1-70B-Instruct"

else:
    model_path = '/mount/shared/Models/Llama-3.1-Nemotron-Nano-8B-v1'

model_path = "/mount/point/veith/Models/Llama-3.1-Nemotron-Nano-8B-v1" # '/mount/point/veith/Models/Llama-3.1-70B-Instruct'

max_memory = {0: "60GB"}# if answ_embedding.lower() in ["y", "yes", "ye", "ja"] else None #Nutzen des max_memory Mappings für die GPUs nur falls ein spezifisches Embedding Modell verwendet wird, um so GPU VRAM zu schonen
#Quantisierungskonfiguration, falls diese benötigt wird
quantization = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  # load_in_8bit=True,
                                  ) #Quantisierungkonfiguration für 4-Bit oder 8-Bit Quantisierung

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left') #Initialisieren des Tokenizers
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             # device='cuda:0'
                                             device_map="auto", #Initialisieren des Modells #Idealfall für Parallelisierung auf GPUs #Auswahl aus ["auto", "balanced", "balanced_low_0", "sequential"]
                                             #max_memory= max_memory #max memory falls benötigt und andere GPUs in Nutzung
                                             quantization_config=quantization if model_question == 3 else None,
                                             #attn_implementation="flash_attention_2", 
                                             torch_dtype=torch.bfloat16, #Konfiguration für Anwendung von flash_attention
                                             ) 
decode_kwargs={'skip_special_tokens':True}
streamer = TextStreamer(tokenizer, skip_prompt=True, **decode_kwargs)

print(f"You are now talking to the {model_path.split('/')[-1]} Model.")

# Funktionsdefinition

CHUNK_CONTEXT_LEN = 100 #globale Variable, welche in mehreren Funktionen abgerufen wird

def split_list(lst, group_size): #Hilfsfunktion zum Aufteilen einer Liste in Dreiergruppen ohne Overlap
    return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]

def load_and_split(loader, chunk_size = tokenizer_embedding.model_max_length):

    """
    Function to load and split documents for the RAG environment.
    Output is the split document.
    """
    markdown_separators = [ #Definition von Trennungszeichen welche gesäubert werden sollen beim Aufteilen der Texte
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
    ]
    chunk_context_len = CHUNK_CONTEXT_LEN #maximale Tokenlänge des zusätzlichen Kontexts
    chunk_overlap=int(chunk_size / 8) #Overlap zwischen Chunks soll ein Achtel der Tokens der benachbarten Chunks beinhalten

    #Failsafe falls maximale Kontextfenster Embedding größer als ideal ist
    chunk_size = min(chunk_size, 1024) #maximale Chunklänge in Tokens beträgt 1024 Tokens
    chunk_size = chunk_size - chunk_context_len - 1 #Anzahl maximal codierbarer Tokens des Sprachmodells minus Tokenlänge der Zusammenfassung

    documents = loader.load() #Laden der rohen Dokumentdaten
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer_embedding,
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=markdown_separators,
    )
    splits = text_splitter.split_documents(documents)

    if len(documents) > 0: #logischer Test ob Daten zur Verarbeitung vorliegen
        if 'page' not in documents[0].metadata: #Test ob Metadaten über die Seiten vorliegen
            text_splitter_alt = RecursiveCharacterTextSplitter.from_huggingface_tokenizer( #alternativer Textsplitter ohne overlap, falls eine große Menge Text ohne Seitenengabe in den Metadaten aus document.load() hervorgeht
                tokenizer_embedding,
                chunk_size = chunk_size,
                chunk_overlap=0,
                add_start_index=True,
                strip_whitespace=True,
                #separators=markdown_separators,
            ) #Verzicht auf Overlap um Chunk Konkatenierung zu vereinfachen

            #Hinzufügen des Attributs 'page' zu den splits
            j=0 #Zählvariable, welche die künstliche Seite der Splits des Dokuments angibt
            for i in range(len(splits)):
                page = j #Festlegen der Seite für Split 
                splits[i].metadata['page'] = page # Hinzufügen von dem Attribut 'page' zu dem Dokumentensplit
                if (i+1)%3==0: # +1 weil ansonsten nur der erste Eintrag mit 0 nummeriert wird
                    j+=1

            splits_alt = text_splitter_alt.split_documents(documents) #Splits, welche gruppenweise konkateniert werden, um Pseudoseiten zu erhalten

            #Auslesen der aktuellen Quelle
            source = documents[0].metadata['source']

            #Auslesen aller Seiteninhalte der Chunks und abspeichern dieser als Strings innerhalb einer Liste
            pages = [] #erstellen einer Hilfsvariable, welche nach dem Schleifendurchlauf als pages definiert wird
            for content in splits_alt: #Splits werden aneinandergereiht in einer Hilfsvariable
                pages.append(content.page_content)

            pages = split_list(pages, 3) #Aufteilen einer Liste in Dreiergruppen von Einträgen #Begründung ist, dass eine PDF-Seite zumeist aus 3 Chunks besteht
            pages = [''.join(map(str, content)) for content in pages] #zusammenführen der Datenstrings aus den Dreiergruppen
            
            #Erstellen des finalen Dokuments mit Pseudoseiten
            documents = [] #Initialisierung der finalen Größe, welche von der Funktion ausgegeben wird

            for i, content in enumerate(pages):
                doc = Document(
                    page_content=content,
                    metadata={'source':source, 'page':i}, #hinzufügen der Metadaten source, wobei source innerhalb eines Dokuments konstant bleibt #hinzufügen der Metadaten page
                )
                documents.append(doc)

    return splits, documents

def model_pipe(llm = model, llm_tokenizer = tokenizer, do_sample = True, streamer = streamer, temperature = 0.6, top_p = 0.9,
               max_new_tokens = 1000, penalty_alpha = 0, top_k = 10, output_scores = False, output_attention = False, output_hidden_states = False):

    """Define the model parameters for the text generation"""

    pipe = pipeline(
        "text-generation", model=llm, tokenizer=llm_tokenizer,
        do_sample=do_sample, #Parameter für Erlaubnis Token Scores zu samplen
        streamer = streamer,# dynamische Darstellung der Textgenerierung
        temperature=temperature, #Temperatur ist der Parameter, welcher die Wahrscheinlichkeitsverteilung der Tokengeneration abflacht und non-greedy sampling ermöglicht
        top_p=top_p, #Der Top p Parameter ist die Grenzwertwahrscheinlichkeit dafür, dass ein bestimmtes Token für die Textgenerierung berücksichtigt wird
        max_new_tokens=max_new_tokens, #Maximale Anzahl an neu generierten Tokens
        penalty_alpha=penalty_alpha, #Penalty Alpha implementiert einen Strafterm bei der Suche nach zu generierenden Tokens durch die Degenerierungsstrafe
        top_k=top_k, #Top k bestimmt die Größe des Sets V(k), welches das Set der top-k predictions von der Wahrscheinlichkeitsverteilung des Sprachmodells
        return_dict_in_generate = False,
        output_scores=output_scores, #Ausgabe der untransformierten Wahrscheinlichkeitsscores für jedes neu generierte Token
        output_attentions=output_attention, #Ausgabe der Attentions für jedes Token
        output_hidden_states=output_hidden_states,
        pad_token_id=tokenizer.eos_token_id,
    ) #Definieren der Modellparameter

    model_pipe = HuggingFacePipeline(pipeline=pipe)

    return model_pipe

def format_docs(docs):
    if isinstance(docs[0], tuple): #Prüfung ob tuple vorliegt durch Prüfung eines Stellvertreters docs[0]
        return "\n\n".join(doc[0].page_content for doc in docs)
    else:
        return "\n\n".join(doc.page_content for doc in docs)
    
#######################################################################################################################################################################################################
#Definition von default Größen bei der Kontextualisierung von Dokumenten
#######################################################################################################################################################################################################

#Prompt Definition zur Kontextualisierung von chunks

prompt_contextualize = PromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

detailed thinking off
Your task is to situate a chunk of text within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
<document>
{context}
</document> 
<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the chunk we want to situate within the whole document 
<chunk> 
{input}
</chunk>
Please give a short succinct and concise context to situate this chunk within the overall document. Write at most 3 sentences.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""")

# Änderung des Prompts falls nicht das Modell Llama-3.1-Nemotron-Nano-8B-v1 verwendet wird
if model_path == '/mount/point/veith/Models/Llama-3.1-Nemotron-Nano-8B-v1':
    detailed_thinking = "\ndetailed thinking off"
else:
    detailed_thinking = "" #leere Ausgabe für detailed_thinking, weil andere Modelle nicht darauf trainiert wurden
# Verändern der Prompt Templates gemäß Wunsch nach detailliertem Denken
prompt_contextualize.template = prompt_contextualize.template.replace('\ndetailed thinking off', detailed_thinking)

#Definition des Sprachmodells
llm_context = model_pipe(do_sample=True, #als LLM wird das Sprachmodell mit dem Standardnamen model ausgewählt
                    temperature = 0.6,
                    top_p = 0.9,
                    max_new_tokens = CHUNK_CONTEXT_LEN, #maximale Anzahl zu generierender Tokens gemäß globaler Variable CHUNK_CONTEXT_LEN
                    penalty_alpha = 0,
                    top_k = 10,
                    output_scores = False,
                    output_attention = False,
                    output_hidden_states = False,
                    )

def contextualize_chunks(doc_splits: list, doc_pages: list, prompt: PromptTemplate = prompt_contextualize, llm: HuggingFacePipeline = llm_context):

    """Contextualizes document chunks in relations to their significance inside the whole document via invocation of an LLM."""

    assert hasattr(doc_splits[0], "page_content") & hasattr(doc_pages[0], "page_content"), "Please enter a valid document split"

    #Formulierung der Kontextualisierungskette mittels Langchain
    chunk_contextualizer = RunnableParallel({"context": RunnablePick("context"), "input": RunnablePick("input")}) | prompt | llm
    #Verwendung RunnableParallel, weil zwei unterschiedliche dynamische Eingaben erfolgen sollen
    #Verwendung von RunnablePick weil dadurch spezifische Größe aus einem dict gelesen werden kann

    #Formatierung der Eingaben
    #Laden von einzelnen vollständigen Dokumenten
    documents = pandas.Series(doc_pages) #Laden der rohen Dokumentdaten aufgeteilt nach Seiten #als pandas.Series um Einträge mittels Masken filtern zu können
    chunks = pandas.Series(doc_splits) #Laden der chunks als pandas.Series() für leichtere Weiterverarbeitung der Daten mittels Maskenfiltern

    #Iteration über einzigartige Quellen
    #vorbereiten der benötigten Daten
    document_sources = pandas.Series([documents[i].metadata['source'] for i in range(len(documents))]) #erstellen einer behelfsmäßigen Liste um die Dokumentenquellen mit den Metadaten der Chunks abzugleichen
    document_sources_unique = document_sources.unique().tolist() #Definition der einzigartigen Quellen über welche iteriert werden soll
    chunk_sources = pandas.Series([doc_splits[i].metadata['source'] for i in range(len(doc_splits))]) #erstellen einer behelfsmäßigen Liste um die Quellen der Chunks mit den Dokumentenquellen abzugleichen

    num_pages = 20 #Variable zur Festlegung der Größe des wandelnden Seitenfensters des betrachteten Dokuments
    interval_half = int(num_pages/2) #ganzzahlige Hälfte des erwünschten Betrachtungsfensters des Dokuments, welche zu den unteren/oberen Intervalllimits der Seiten addiert werden kann
    token_pagebreak = "/pagebreakfordoc\n" #Token zur Aufteilung der Dokumentseiten unter Beachtung, dass diese Textsequenz nicht durch Zufall im Dokument vorkommt

    progress_bar = tqdm(range(len(chunks)))

    for source in document_sources_unique:
        #erstellen eines einzelnen großen Dokuments für die aktuelle einzigartige Quelle
        mask = (source == document_sources) #erstellen einer Maske für die Filterung der relevanten Dokumentseiten, welche mit der aktuell betrachteten Quelle übereinstimmen
        whole_document = "" #Initialisierung der Variable für gesamte Dokumente
        n = len(documents[mask]) #Seitenzahl des Dokuments

        for i in range(len(documents[mask])): #Iteration über alle Seiteneinträge der der aktuellen Quelle, welche mittels [mask] abgerufen wird
            whole_document = whole_document + token_pagebreak + documents[mask].iloc[i].page_content #Seitenweises erweitern des aktuellen Strings von whole_document
        #gesamtes Dokument liegt nun als einzelner String vor um and das LLM übergeben zu werden
        whole_document_pagebreaks = whole_document.split(token_pagebreak)[1:] #Aufteilen des Dokuments in Seiten nach pagebreak Token mit auslassen des ersten Eintrag, weil dieser ein leerer String ist
    
        #Laden von einzelnen Dokumenten Chunks der aktuellen Quelle
        mask = (source == chunk_sources) #erstellen einer Maske für die Filterung der Chunks, welche mit der aktuell betrachteten Quelle übereinstimmen

        for i, chunk in enumerate(chunks[mask]): #Iteration über, nach der aktuellen Quelle zugehörigen, Chunks
            current_page = chunk.metadata['page'] #Ausgabe der Seite des aktuellen Chunks #Beginn der Seitenzählung des Dokuments bei 0
            #Erstellen des Dokuments
            #Fallunterscheidung für Implementierung eines num_pages großen Betrachtungsfensters
            limit_lower = 0 if current_page < interval_half else min(current_page-interval_half, n-num_pages)
            limit_upper = max(current_page+interval_half, num_pages) if current_page <= n-interval_half else n
            whole_document_chunks = whole_document_pagebreaks[limit_lower:limit_upper] #Indizierung, sodass nur Seiten innerhalb des Betrachtungsfesnsters an den Prompt übergeben werden
            whole_document_chunks_string = "" #Hilfsgröße, welche Stück für Stück mit den im Intervall betrachteten Texten befüllt wird
            for text in whole_document_chunks: #Durchlauf über alle Dokumentseiten innerhalb des Betrachtungsfensters
                whole_document_chunks_string = whole_document_chunks_string + "\n" + text #Hinzufügen von relevantem Text zu der Variable
            whole_document_chunks_string = whole_document_chunks_string.replace(token_pagebreak, '') #entfernen des pagebreak Tokens
            
            context_chunk = chunk_contextualizer.invoke({"context": whole_document_chunks_string, "input": chunk.page_content}) #Kontextualisieren der Chunks mittels LLM
            context_chunk = context_chunk.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1] #parsen des LLM Outputs
            chunks[mask].iloc[i].page_content = context_chunk + ";" + chunk.page_content #ersetzen der alten Chunks mit den neuem LLM generierten Kontext als Präfix innerhalb Variable chunks
            progress_bar.update(1)
            #splits[i].page_content wird durch Beziehung zu Variable chunks automatisch rückwirkend ersetzt

    return doc_splits #Ausgabe der Dokumentensplits nachgestellt an load_and_split()

def save_csv_data(doc_splits: list, save_dir: str):
    """Saves contextualized chunks inside a dataframe for further usage."""
    #erstellen eines Dataframes aus str page_content und dicts für Metadaten
    df_splits = pandas.DataFrame({"content":[doc_splits[i].page_content for i in range(len(doc_splits))],
                                "source":[doc_splits[i].metadata['source'] for i in range(len(doc_splits))],
                                "page": [doc_splits[i].metadata['page'] for i in range(len(doc_splits))],
                                "start_index": [doc_splits[i].metadata['start_index'] for i in range(len(doc_splits))]})
    #speichern des Dataframes
    directory = save_dir + '.csv' #erweitern des save_dir um die Endung .csv zum Abspeichern als eigene Datei speziell für BM25 retrieval
    return df_splits.to_csv(directory) #abspeichern der Datei in ein auslesbares Format mittels load_csv_data

def load_csv_data(load_dir:str):
    '''Loads .csv tables as Dataframes and formats the output as a dataframe, text data and metadata.'''
    #laden des Dataframes
    if load_dir.endswith(".csv"): #Test ob load_dir auf .csv endet oder nicht
        directory = load_dir
    else:
        directory = load_dir + '.csv'
    
    df = pandas.read_csv(directory)
    if 'Unnamed: 0' in df.columns: #Test ob Speicherartefakt in Form Spalte 'Unnamed: 0' existiert
        df = df.drop('Unnamed: 0', axis=1) #entfernen der unerwünschten Datenspalte
    texts = df['content']

    #umwandeln der Metadaten in lesbares Format
    metadata = [] #initialisieren der Metadaten als leere Liste
    for i, src in enumerate(df['source']):
        dic = {'source': src, "page": df['page'][i], "start_index": df['start_index'][i]}
        metadata.append(dic)
    return df, texts, metadata


def rag_env(directory: str = None, file_paths: str = None, data_path: str = None, embedding = model_embedding, vector_store_type = FAISS, save_dir: str = None, contextualization=True):

    """
    Function to create a vectorized RAG environment. Input consists of either the folder path or a list of specific file paths. The output is the retriever for the vector store of the documents.
    """
    if isinstance(file_paths, str): #Umwandlung file_path in list, falls ein str eingegeben wird
        file_paths = [file_paths]

    assert isinstance(directory, str) | isinstance(file_paths, list) | isinstance(data_path, str), "Either enter a valid folder or data path as string or enter file paths compiled into a single list object."
    #Überprüfen ob mindestens eine korrekte Eingabe für die Variablen directory oder file-paths vorliegt

    docs = [] #Initialisierung der Liste, welche die Dokumenteninformationen beinhalten wird

    if isinstance(directory, str): #Eingängliche Überprüfung ob directory angegeben wurde bevor die Daten ausgelesen werden

        print("Contextualizing document chunks...")

        pdf_loader = DirectoryLoader(directory, glob='./*.pdf', loader_cls=PyPDFLoader)
        splits, pages = load_and_split(pdf_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        txt_loader = DirectoryLoader(directory, glob='./*.txt', loader_cls=TextLoader)
        splits, pages = load_and_split(txt_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        word_loader = DirectoryLoader(directory, glob='./*.docx', loader_cls=Docx2txtLoader)
        splits, pages = load_and_split(word_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        csv_loader = DirectoryLoader(directory, glob='./*.csv', loader_cls=CSVLoader)
        splits, pages = load_and_split(csv_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        json_loader = DirectoryLoader(directory, glob='./*.json', loader_cls=JSONLoader)
        splits, pages = load_and_split(json_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list
    


    if isinstance(file_paths, list): #Eingängliche Überprüfung ob file_paths angegeben wurden bevor die Daten ausgelesen werden

        for file in file_paths: #Iterieren über alle angegebenen Dateipfade
            
            #Fallunterscheidung der Loader abhängig von Dateiendung #keine Änderung wie beim directory notwendig, weil Dateien nur bei expliziter Nennung eingelesen werden
            if file.endswith('.pdf'):
                pdf_loader = PyPDFLoader(file)
                splits, pages = load_and_split(pdf_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)

            elif file.endswith('.txt'):
                txt_loader = TextLoader(file)
                splits, pages = load_and_split(txt_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)

            elif file.endswith('.docx'):
                word_loader = Docx2txtLoader(file)
                splits, pages = load_and_split(word_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)
            
            elif file.endswith('.csv'):
                csv_loader = CSVLoader(file, csv_args={'delimiter':','})
                splits, pages = load_and_split(csv_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)
            
            elif file.startswith('https://'):
                web_loader = WebBaseLoader(file)
                splits, pages = load_and_split(web_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)
        
    #docs #Zusammengestellte Dokumente als Liste

    # Vektorisieren der Dokumente
    if isinstance(data_path, str): #Eingängliche Überprüfung ob data_path angegeben wurde bevor die Daten ausgelesen werden
        df, texts, metadata = load_csv_data(data_path)
        vector_store = vector_store_type.from_texts(texts=texts.astype(str), embedding=embedding, metadatas=metadata) #FAISS Vektorisierung aus Datensatz

        #erstellen einer Liste von docs
        docs = [] #Initialisierung der finalen Größe, welche von der FUnktion ausgegeben wird
        for i, content in enumerate(df['content']):
            doc = Document(
                page_content=content,
                metadata={'source':df['source'].iloc[i], 'page':df['page'].iloc[i], 'start_index':df['start_index'].iloc[i]}, #hinzufügen der Metadaten source, wobei source innerhalb eines Dokuments konstant bleibt #hinzufügen der Metadaten page
            )
            docs.append(doc)
    else:
        vector_store = vector_store_type.from_documents(docs, embedding) #FAISS Vektorisierung

    # Aufräumen
    gc.collect() #Befreien der GPU Memory
    torch.cuda.empty_cache() #Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi

    #Speichern des Vector Stores
    if save_dir != None:
        vector_store.save_local(save_dir)
        save_csv_data(docs, save_dir=save_dir) #abspeichern der Chunks mit Metadaten in einem Dataframe

    return vector_store, docs #Ausgabe des Vektordatenspeichers und der Dokumentensplits vorbereitend für die Definition der Retriever


def load_retriever_vector_store(vector_store_path: str=None, vector_store=None, doc_splits=None, embedding= model_embedding, search_type = "similarity", num_k = 3, vector_store_type=FAISS, retrieval_scores=False):
    """Loads a vector store and returns a retriever"""

    #Prüfung ob ein valider input vorliegt
    assert (hasattr(vector_store, "as_retriever")) | (isinstance(vector_store_path, str)), "Please enter the path to a vector_store or the vector store itself"

    search_types = ["similarity", "mmr"] #mögliche search_types #"semantic" als eigenes Suchkriterium für Spezialfall

    assert search_type in search_types, f"Please enter a valid search type present in {search_types}"

    if isinstance(vector_store_path, str): #Fallunterscheidung falls vector_store_path eingegeben wird
    
        # Laden des verwiesenen Vektorspeichers
        vector_store = vector_store_type.load_local(folder_path=vector_store_path,
                                                    embeddings=embedding,
                                                    index_name="index",
                                                    allow_dangerous_deserialization=True)
    
    #Prüfung ob vector_store als retriever verwendet werden kann
    assert hasattr(vector_store, "as_retriever"), "Please enter a vector store that can be used for document retrieval"

    #Definition des retrievers
    #Fallunterscheidung je nach Konfiguration in dict()
    search_configs = {"similarity": {"True": RunnableLambda(vector_store.similarity_search_with_score).bind(k=num_k),
                                     "False": RunnableLambda(vector_store.similarity_search).bind(k=num_k)},
                      "mmr": {"True": RunnableLambda(vector_store.max_marginal_relevance_search_with_score_by_vector).bind(k=num_k),
                                     "False": RunnableLambda(vector_store.max_marginal_relevance_search).bind(k=num_k)},
                      }
    retriever_semantic = search_configs[search_type][f"{retrieval_scores}"] #verweis auf die korrekte erwünschte Suchkonfiguration anhängig von Eingabeparametern
    #Default Definition des retrievers
    # retriever = vector_store.as_retriever(search_kwargs={"search_type":search_type, "k":num_k}) #Definition retriever #Definition Suchparameter retriever gemäß search_type und num_k
    
    #Fallunterscheidung je nachdem, ob auf vorhandene Wissensbasis zurückgegriffen wird oder auf neu generierte
    if vector_store_path!=None:
        #Für den Fall, dass vector_store_path angegeben wurde
        df, texts, metadata = load_csv_data(vector_store_path)
        retriever_bm25 = BM25Retriever.from_texts(texts.astype(str), metadatas=metadata)#einesen der .csv Datei als Dataframe und UMwandlung mittels Funktion in lesbares dict damit .from_texts() mit Metadaten funktioniert
        retriever_bm25.k = num_k
    else: #Für den Fall, dass vector_store angegeben wurde
        retriever_bm25 = BM25Retriever.from_documents(doc_splits)
        retriever_bm25.k = num_k

    retriever = EnsembleRetriever(retrievers=[retriever_semantic, retriever_bm25],
                                       weights=[0.5, 0.5]) #hybrider Retriever, welcher letzenendes für Dokumentensuche verwendet wird
    
    return retriever, vector_store


def rag_env_expand(vector_store_path: str, directory: str = None, file_paths: str = None, save_dir: str=None, embedding = model_embedding,
                   vector_store_type = FAISS, contextualization=True):
    """Expand the given vector store with new documents."""

    if isinstance(file_paths, str): #Umwandlung file_path in list, falls ein str eingegeben wird
        file_paths = [file_paths]

    assert isinstance(directory, str) | isinstance(file_paths, list), "Either enter a valid folder path as string or enter file paths compiled into a single list object."
    #Überprüfen ob mindestens eine korrekte Eingabe für die Variablen directory oder file-paths vorliegt

    docs = [] #Initialisierung der Liste, welche die Dokumenteninformationen beinhalten wird

    print("Contextualizing document chunks...")

    if isinstance(directory, str): #Eingängliche Überprüfung ob directory angegeben wurde bevor die Daten ausgelesen werden

        pdf_loader = DirectoryLoader(directory, glob='./*.pdf', loader_cls=PyPDFLoader)
        splits, pages = load_and_split(pdf_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        txt_loader = DirectoryLoader(directory, glob='./*.txt', loader_cls=TextLoader)
        splits, pages = load_and_split(txt_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        word_loader = DirectoryLoader(directory, glob='./*.docx', loader_cls=Docx2txtLoader)
        splits, pages = load_and_split(word_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        csv_loader = DirectoryLoader(directory, glob='./*.csv', loader_cls=CSVLoader)
        splits, pages = load_and_split(csv_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list

        json_loader = DirectoryLoader(directory, glob='./*.json', loader_cls=JSONLoader)
        splits, pages = load_and_split(json_loader) #aufteilen der Dokumente
        if (len(splits) > 0) & (contextualization==True): #überspringen der folgenden Schritte, wenn keine Dokumente des gewünschten Datentyps im angegebenen directory vorhanden sind oder keine Kontextualisierung erwünscht ist
            splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
        docs.extend(splits) #Aufnehmen der gespaltenen Dokumentenabschnitte in list
    


    if isinstance(file_paths, list): #Eingängliche Überprüfung ob file_paths angegeben wurden bevor die Daten ausgelesen werden

        for file in file_paths: #Iterieren über alle angegebenen Dateipfade
            
            #Fallunterscheidung der Loader abhängig von Dateiendung #keine Änderung wie beim directory notwendig, weil Dateien nur bei expliziter Nennung eingelesen werden
            if file.endswith('.pdf'):
                pdf_loader = PyPDFLoader(file)
                splits, pages = load_and_split(pdf_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)

            elif file.endswith('.txt'):
                txt_loader = TextLoader(file)
                splits, pages = load_and_split(txt_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)

            elif file.endswith('.docx'):
                word_loader = Docx2txtLoader(file)
                splits, pages = load_and_split(word_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)
            
            elif file.endswith('.csv'):
                csv_loader = CSVLoader(file, csv_args={'delimiter':','})
                splits, pages = load_and_split(csv_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)
            
            elif file.startswith('https://'):
                web_loader = WebBaseLoader(file)
                splits, pages = load_and_split(web_loader)
                if contextualization==True: #Auslassen Kontextualisierung falls diese unerwünscht ist
                    splits = contextualize_chunks(doc_splits=splits, doc_pages=pages) #kontextualisieren der Dokumentenchunks
                docs.extend(splits)
    #docs #Zusammengestellte Dokumente als Liste

    vector_store = vector_store_type.load_local(folder_path=vector_store_path,
                                    embeddings=embedding,
                                    allow_dangerous_deserialization=True)
    vector_store.add_documents(documents=docs) #Befehl um neue Dokumente dem Vector Store hinzuzufügen
    df, texts, metadata = load_csv_data(vector_store_path)
    #Hinzufügen von neuen Zeilen in den Dataframe
    df_splits = pandas.DataFrame({"content":[docs[i].page_content for i in range(len(docs))],
                                "source":[docs[i].metadata['source'] for i in range(len(docs))],
                                "page": [docs[i].metadata['page'] for i in range(len(docs))],
                                "start_index": [docs[i].metadata['start_index'] for i in range(len(docs))]}) #erstellen Dataframe bestehend aus neuen Dokumenten
    #verbinden der Datensätze zu gräßerem Datensatz
    df = pandas.concat([df, df_splits], ignore_index = True)
    # Aufräumen
    gc.collect() #Befreien der GPU Memory
    torch.cuda.empty_cache() #Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi

    #Speichern des Vector Stores
    if save_dir != None:
        assert os.path.exists(save_dir), "Please enter a valid path"
        vector_store.save_local(save_dir)
        dir_df = save_dir + '.csv'
        df.to_csv(dir_df) #abspeichern der Datei in ein auslesbares Format mittels load_csv_data #Änderung bedingt durch Ergänzung des vorhandenen Dataframes

    return vector_store, docs

##################################################################################################################################################################################################
#Textgenerierungsfunktionen
##################################################################################################################################################################################################

prompt_query = PromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

detailed thinking off
Your task is to convert the Input into database search terms with natural language.
Format the answer like this:
SEARCH: [search]

Given a question or input, return a single search term optimized to retrieve the most relevant results from a search engine.
If there are acronyms or words you are not familiar with, do not try to rephrase them.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Input: {input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""")

# Änderung des Prompts falls nicht das Modell Llama-3.1-Nemotron-Nano-8B-v1 verwendet wird
prompt_query.template = prompt_query.template.replace('\ndetailed thinking off', detailed_thinking)

@chain
def query_parser(query: str):
    """Parses the output of the query analyzer, so it only returns the search_query"""
    new_query = query.split(sep="SEARCH: ", maxsplit=-1)[-1]
    new_query = new_query.split(sep="<|eot_id|>")[0]
    task_description = "Given a web search query, retrieve relevant passages that are relevant to the query."
    task = f'Instruct: {task_description}\nQuery: {new_query}' #Ausgabe für Instruction trainierte Embedding Modelle #new_query fungiert als geeigneter Suchbegriff und task_description beschreibt den Suchkontext
    # Struktur der Task ist modellspezifisch anzupassen
    # print(task) #Für Debugging
    return task #new_query


def rag_gen(instruction : str, prompt: ChatPromptTemplate, retriever : VectorStoreRetriever, chat_history, extra = dict(), query_analysis = True,
            do_sample = True, streamer = streamer, streamer_query=None, temperature = 0.6, top_p = 0.9,
            max_new_tokens = 1000, penalty_alpha = 0.3, top_k = 10, output_scores = False, output_attention = False, output_hidden_states = False, retrieval_scores = False,
            print_sources = True,):

    #Überprüfen der Eingaben
    assert isinstance(instruction, str), "Please enter the instruction in form of a string"

    assert isinstance(extra, dict), "Please enter the extra variables in form of string(s) inside a dictionary, with the placeholder variable inside the prompt as keys and the corresponding strings as values"

    #Definition der zu verwendenden Modellpipeline

    llm = model_pipe(do_sample=do_sample,
                     streamer=streamer,
                     temperature = temperature,
                     top_p = top_p,
                     max_new_tokens = max_new_tokens,
                     penalty_alpha = penalty_alpha,
                     top_k = top_k,
                     output_scores = output_scores,
                     output_attention = output_attention,
                     output_hidden_states = output_hidden_states,
                     )


    rag_chain = (
        #{"context": retriever | format_docs, "input": RunnablePassthrough()} 
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))# Formatieren des eingegebenen Kontexts
        #format_docs formatiert die Docs, sodass nur konkatenierter page_content wiedergegeben wird und beschränkt die Textgenerierung
        | prompt
        | llm
        | StrOutputParser()
    )

    if query_analysis == False:
        rag_chain_dict = {"context": retriever, "chat_history": chat_history, "input": RunnablePassthrough()}
        if len(extra) >= 1:
            rag_chain_dict["input"] = RunnablePick('input') #Änderung der Variable 'input' von RunnablePassthrough in RunnablePick, weil bei invoke ein dict übergeben wird mit mehreren Größen, anstatt nur einer Variable
            for i in extra.keys():
                rag_chain_dict[i] = RunnablePick(i)

        rag_chain_with_source = RunnableParallel(
            rag_chain_dict
        ).assign(answer=rag_chain)
    
    else:
        #Zunächst Definition der LLM Pipeline zur Generierung der Suchquery
        llm_query = model_pipe(do_sample=False, #Deaktivieren des Samplings für deterministische Query-Generierung
                     streamer=streamer_query,
                     temperature = None,
                     top_p = None,
                     max_new_tokens = 100,
                     top_k = 10,
                     )
        
        query_analyzer = {"input": RunnablePassthrough()} | prompt_query | llm_query | query_parser | retriever #Definition der neuen retrieval chain, welche Inhalte gemäß der neuen Query heraussucht

        rag_chain_dict = {"context": query_analyzer, "chat_history": chat_history, "input": RunnablePassthrough()}
        if len(extra) >= 1:
            rag_chain_dict["input"] = RunnablePick('input') #Änderung der Variable 'input von RunnablePassthrough in RunnablePick, weil bei invoke ein dict übergeben wird mit mehreren Größen, anstatt nur einer Variable
            for i in extra.keys():
                rag_chain_dict[i] = RunnablePick(i)

        rag_chain_with_source = RunnableParallel(
            rag_chain_dict #Standard: Eingabeschema für 2 Eingabevariablen
        ).assign(answer=rag_chain)

    if len(extra) >= 1:
        for i in extra.keys():
            rag_chain_dict[i] = extra[i]
        rag_chain_dict['input'] = instruction
        gen = rag_chain_with_source.invoke(rag_chain_dict) 
    else:
        gen = rag_chain_with_source.invoke(instruction) 
    #Ablauf: Zunächst wird die Frage "question" für die Variable {input} in den Prompt eingegeben
    #Daraufhin lädt der Retriever Dokumente aus dem Vector Store der Variable vectorstore mit der höchsten Kosinusähnlichkeit (default)
    #Auf die beschaffenen Dokumente wird die Funktion format_docs() angewandt -> Die Dokumente werden konkateniert
    #Diese Dokumente werden in die Variable {context} von dem Prompt geladen
    #Falls zusätzliche Variablen in dem verwendeten Prompt vorliegen werden diese aus der Variable 'extra' herausgelesen
    #Die Variable prompt stellt den Kontext und das Eingabeschema bereit mit den 
    #Daraufhin wird für den gegebenen Kontext die Antwort des Sprachmodell über die Pipeline generiert
    #Zuletzt wird durch StrOutputParser das beste Ergebnis aus den generierten Tokens bezogen

    #Parsing der Ausgabe

    response = gen["answer"].split("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n") #aufteilen string sodass nur Modellausgabe ohne Kontext stattfindet aber mit Frage #geeigneter Split ist promptabhängig
    response = response[-1] #Definition der Antwort als das letzte Listenelement, welches den String nach dem Separator "</context>\n\n" darstellt
    response_question = response.split("<|eot_id|>")[0] #Speichern des Question Strings, welcher nach dem Token <|eot_id|> endet
    response_answer = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")[-1] #Trennen nach Prompt Tokens, welche den Antwortbereich für das Modell signalisieren
    response = response_question + response_answer #Zusammenfügen der Frage und Antwort
    #print(response[0])

    if print_sources == True: #Fals die Quellenangaben mit ausgegeben werden sollen
        for source in gen['context']:
            print("Sources:")
            if isinstance(source, tuple): #Fallunterscheidung falls retrieval_scores ausgegeben werden -> dies resultiert in Ausgabe von tuple für source mit source[1] als retrieval score
                print(source[0].metadata)
                print(f"Retrieval Score: {source[1]}")
            else:
                print(source.metadata)

    # Aufräumen
    gc.collect() #Befreien der GPU Memory
    torch.cuda.empty_cache() #Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi

    return gen, response_answer #Ausgabe der response answer zur Weitergabe an die Chat History