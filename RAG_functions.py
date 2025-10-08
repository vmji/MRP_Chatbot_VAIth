import os #Zugriff auf Dateien für RAG
os.environ['USER_AGENT'] = 'myagent'
import sys

from typing import Union, List
import time
import torch
import gc #GPU Memory Optimierung
import pandas
from tqdm.auto import tqdm #Fortschrittsbalken für Promptausgaben

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline, BitsAndBytesConfig, Mistral3ForConditionalGeneration #Funktionalität Hugging Face Modelle
# import accelerate #Für Anwendung der Berechnungen auf GPUs
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate #Prompt Templates
from langchain_core.prompt_values import StringPromptValue
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings #Nutzung von Hugging Face Modellen

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader, TextLoader, Docx2txtLoader #Laden von PDF Dateien, Webseiten, Pandas Datensätzen, .txt Dateien, Word Dokumente
from langchain_community.document_loaders.csv_loader import CSVLoader # Laden von .csv Dateien
from langchain_community.document_loaders.json_loader import JSONLoader # Laden von .json Dateien
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_message_histories import ChatMessageHistory #Chat History

#from langchain_community.document_loaders import GoogleDriveLoader, OneDriveFileLoader #nur falls es unbedingt notwendig, weil Zugriff auf Daten sehr schwierig ist

#Indizieren des Vektorraums mit Dokumenten
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Funktionen Quellenangaben
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, chain, RunnablePick

FILE_PATH = sys.argv[1] #Pfad zu den Dateien, welche verarbeitet werden sollen

#Konfiguration der Modelle

print("Initializing embedding model...")
model_kwargs = {'device': 'cuda:3',
                "trust_remote_code": "True",
                "model_kwargs": {"dtype": "bfloat16",}} #Argumente für Embedding Modell
encode_kwargs = {'normalize_embeddings': True} #Argumente für Codierung der Textsequenzen
model_embedding_path = "/mount/point/veith/Models/multilingual-e5-large-instruct"
model_embedding = HuggingFaceEmbeddings(model_name=model_embedding_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs) #auslagern der Rechenleistung auf GPU
#wird benutzt falls eigene Dateien hochgeladen wurden, aber das standardmäßige Embeddingmodell verwendet wird
tokenizer_embedding = AutoTokenizer.from_pretrained(model_embedding_path)

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

max_memory = {0: "28GB", 1: "28GB", 2: "28GB", 3: "28GB"}# if answ_embedding.lower() in ["y", "yes", "ye", "ja"] else None #Nutzen des max_memory Mappings für die GPUs nur falls ein spezifisches Embedding Modell verwendet wird, um so GPU VRAM zu schonen
#Quantisierungskonfiguration, falls diese benötigt wird
quantization_4bit = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  ) #Quantisierungkonfiguration für 4-Bit oder 8-Bit Quantisierung
quantization_8bit = BitsAndBytesConfig(load_in_8bit=True,) #Quantisierungkonfiguration für 8-Bit Quantisierung

# Frage nach zu verwendendem Modell
print("Which text generation model do you want to use?")
print("1.\tLlama-3.1-8B-Instruct")
print("2.\tLlama-3.1-70B-Instruct (4bit quantized)")
print("3.\tQwen3-14B")
print("4.\tMagistral-Small-2509 (8bit quantized)")
model_question = int(input("Please answer by typing in the corresponding model number: "))

# Dictionary der Modellkonfigurationen
MODEL_CONFIGS = {
    1: {
        "name": "Llama-3.1-8B-Instruct",
        "path": "/mount/point/veith/Models/Llama-3.1-8B-Instruct",
        "model_class": AutoModelForCausalLM,
        "quantization": None,
        "special_tokenizer_args": {"padding_side": "left"},
        "special_model_args": {},
        "token_family": "llama",
        "reasoning": False
    },
    2: {
        "name": "Llama-3.1-70B-Instruct",
        "path": "/mount/point/veith/Models/Llama-3.1-70B-Instruct",
        "model_class": AutoModelForCausalLM,
        "quantization": quantization_4bit,
        "special_tokenizer_args": {"padding_side": "left"},
        "special_model_args": {},
        "token_family": "llama",
        "reasoning": False
    },
    3: {
        "name": "Qwen3-14B",
        "path": "/mount/point/veith/Models/Qwen3-14B",
        "model_class": AutoModelForCausalLM,
        "quantization": None,
        "special_tokenizer_args": {"padding_side": "left"},
        "special_model_args": {},
        "token_family": "qwen",
        "reasoning": True
    },
    4: {
        "name": "Magistral-Small-2509",
        "path": "/mount/point/veith/Models/Magistral-Small-2509",
        "model_class": Mistral3ForConditionalGeneration,
        "quantization": quantization_8bit,
        "special_tokenizer_args": {"tokenizer_type": "mistral", "use_fast": False},
        "special_model_args": {},
        "token_family": "mistral",
        "reasoning": True
    }
}

def initialize_model(model_choice: int):
    """
    Initialize model and tokenizer based on user choice.
    
    Args:
        model_choice (int): Model number from user selection
        
    Returns:
        tuple: (model, tokenizer, model_path)
    """
    if model_choice not in MODEL_CONFIGS:
        print(f"Invalid model choice {model_choice}. Defaulting to Qwen3-14B.")
        model_choice = 4
    
    config = MODEL_CONFIGS[model_choice]
    model_path = config["path"]
    
    print(f"Initializing {config['name']}...")
    
    # Initialisieren tokenizer
    tokenizer_args = config["special_tokenizer_args"].copy()
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
    
    # Standard Modell args
    model_args = {
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
        "dtype": torch.bfloat16,
    }
    
    # Quantisierung, falls benötigt
    if config["quantization"] is not None:
        model_args["quantization_config"] = config["quantization"]
    
    # Hinzufügen modellspezifischer args
    model_args.update(config["special_model_args"])
    
    # Initialisieren des Modells mit der korrekten Klasse
    model = config["model_class"].from_pretrained(model_path, **model_args)
    
    # Anwenden des eval Modus, falls spezifiziert
    if config["special_model_args"].get("eval_mode", False):
        model = model.eval()
    
    return model, tokenizer, model_path

# Modellinitialisierung
model, tokenizer, model_path = initialize_model(model_question)

decode_kwargs={'skip_special_tokens':True}
streamer = TextStreamer(tokenizer, skip_prompt=True, **decode_kwargs)

time.sleep(2)
print(f"You are now talking to the {MODEL_CONFIGS[model_question]['name']} Model.")

TOKEN_FAMILIES = {
    "llama": {
        "SYSTEM_START": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
        "USER_START": "<|start_header_id|>user<|end_header_id|>",
        "ASSISTANT_START": "<|start_header_id|>assistant<|end_header_id|>",
        "TURN_END": "<|eot_id|>",
        "REASONING_OFF": "",
        "REASONING_ON": "",
        "REASONING_END": "</think>"
    },
    "qwen": {
        "SYSTEM_START": "<|im_start|>system",
        "USER_START": "<|im_start|>user",
        "ASSISTANT_START": "<|im_start|>assistant",
        "TURN_END": "<|im_end|>",
        "REASONING_OFF": "<|im_start|>assistant\n<think>\n\n</think>",
        "REASONING_ON": "<|im_start|>assistant",
        "REASONING_END": "</think>"
    },
    "mistral": {
        "SYSTEM_START": "<s>[SYSTEM_PROMPT]",
        "USER_START": "[/SYSTEM_PROMPT][INST]",
        "ASSISTANT_START":"[/INST]</s>",
        "TURN_END": "", # kein eigenes Turn Token, wird mittels user und assistant Token definiert
        "REASONING_OFF": "[/INST]",
        "REASONING_ON": "[/INST]First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.",
        "REASONING_END": "[/THINK]"
    }
}

# Übersetzung modellspezifischer Tokens mit default reasoning off
def translate_tokens(prompt: str, model_choice: int) -> str:
    """Translate modellspecific tokens in a prompt to the tokens of the specified model."""
    if model_choice not in MODEL_CONFIGS:
        model_choice = 4 # Default to Qwen3-14B if invalid choice
    config = MODEL_CONFIGS[model_choice] # Beschaffen der Modellkonfiguration
    token_family = config["token_family"] # Beschaffen der Modellfamilie
    special_tokens = TOKEN_FAMILIES[token_family] # Abrufen der Spezialtokens der Modellfamilie
    # Definieren der speziellen Strings
    parsing_string = special_tokens["ASSISTANT_START"]
    parsing_string_user = special_tokens["USER_START"]

    # Ersetzen der Tokens im Prompt
    for key in special_tokens:
        prompt = prompt.replace(key, special_tokens[key])
    
    # Ausnahmefall, dass es sich um ein Reasoning-Modell handelt
    if config["reasoning"] == True:
        prompt = prompt.replace(special_tokens["ASSISTANT_START"], special_tokens["REASONING_OFF"]) # default abschalten vom Reasoning
    
    return prompt, parsing_string, parsing_string_user

# Promptdefinition

#Prompt Definition zur Kontextualisierung von chunks

prompt_contextualize = PromptTemplate.from_template("""SYSTEM_START

Your task is to situate a chunk of text within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
<document>
{context}
</document> 
TURN_ENDUSER_START

Here is the chunk we want to situate within the whole document 
<chunk> 
{input}
</chunk>
Please give a short succinct and concise context to situate this chunk within the overall document. Write at most 3 sentences.TURN_END
ASSISTANT_START
""")

# Prompt zum erstellen einer Suchquery für RAG-Anwendungen

prompt_query = PromptTemplate.from_template("""SYSTEM_START

Your task is to convert the Input into database search terms with natural language.
Format the answer like this:
SEARCH: [search terms]
STRICT FORMAT REQUIREMENTS:
1. Start your response with exactly "SEARCH: " (including the space)
2. Follow with the search terms (single line, no explanations)
3. Do NOT add any text before or after the required format

Given a question or input, return a single search term optimized to retrieve the most relevant results from a search engine.
If there are acronyms or words you are not familiar with, do not try to rephrase them.

TURN_ENDUSER_START

Input: {input}TURN_END
ASSISTANT_START
""")

# Übersetzen der Prompts in die modellspezifische Schreibweise
for prompt in [prompt_contextualize, prompt_query]:
    prompt.template, parsing_string, parsing_string_user = translate_tokens(prompt.template, model_question)

# Funktionsdefinition

CHUNK_CONTEXT_LEN = 100 #globale Variable, welche in mehreren Funktionen abgerufen wird

# Übersetzung modellspezifischer Tokens, falls nicht default llama Modell ausgewählt wird
if "llama" not in model_path.lower():

    if "qwen" in model_path.lower(): # Fall dass ein Qwen Modell verwendet wird
        llama_qwen = {
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>": "<|im_start|>system", #Systemprompt
            "<|start_header_id|>user<|end_header_id|>": "<|im_start|>user", #Userprompt
            "<|start_header_id|>assistant<|end_header_id|>": "<|im_start|>assistant", # Beginn Gesprächsanteil Assistent
            "<|eot_id|>": "<|im_end|>", # Gesprächsrundentoken
        }
        llama_translator = llama_qwen # Definition des zu verwendenden dict zum Übersetzen der modellspezifischen Tokens
    
    #Definition der Parsing Strings
    parsing_string = llama_translator['<|start_header_id|>assistant<|end_header_id|>']
    parsing_string_user = llama_translator['<|start_header_id|>user<|end_header_id|>']
    parsing_chat_turn_token = llama_translator['<|eot_id|>']
else:
    #Definition der Parsing Strings
    parsing_string = '<|start_header_id|>assistant<|end_header_id|>'
    parsing_string_user = '<|start_header_id|>user<|end_header_id|>'
    parsing_chat_turn_token = '<|eot_id|>'

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

    #Failsafe falls maximale Kontextfenster Embedding größer als ideal ist
    chunk_size = min(chunk_size, 1024) #maximale Chunklänge in Tokens beträgt 1024 Tokens
    chunk_size = chunk_size - chunk_context_len - 1 #Anzahl maximal codierbarer Tokens des Sprachmodells minus Tokenlänge der Zusammenfassung
    chunk_overlap=int(chunk_size / 8) #Overlap zwischen Chunks soll ein Achtel der Tokens der benachbarten Chunks beinhalten

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

default_config = {
    "do_sample": True, 
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": 1000,
    "top_k": 10,
}
# Speichereffiziente Lösung der Pipelinedefinition
class PipelineManager:
    def __init__(self, model, tokenizer, initial_params=None,):
                 #streamer = None):
        self.model = model
        self.model_path = model.name_or_path
        self.tokenizer = tokenizer
        self.current_params = initial_params
        self.pipeline = None
        
        # Modellspezifische Fixes
        # Fix tokenizer padding für Llama Modelle
        if 'llama' in self.model_path.lower() and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if initial_params!=None:
            self.get_pipeline(**initial_params)
    
    def get_pipeline(self, **params):
        # Neudefinition der Pipeline findet nur statt wenn sich die Parameter ändern
        if self.current_params is None:
            self.current_params = params.copy()
        if self.current_params != params or self.pipeline is None:
            # Abändern der neuen Parameter, während alte Parameter beibehalten werden
            for parameter in params.keys():
                if parameter in self.current_params:
                    self.current_params[parameter] = params[parameter]
            # Hinzufügen von Parametern, die vorher nicht gesetzt waren
            for parameter in params.keys():
                if parameter not in self.current_params:
                    self.current_params[parameter] = params[parameter]

            pipe = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                # streamer = self.streamer,
                **self.current_params
            )
            self.pipeline = HuggingFacePipeline(pipeline=pipe)
        return self.pipeline

# Alternative zu model_pipe()
# Function to wrap tokenization, model.generate() and decoding, so that it can be reused easily
@chain
def model_pipe_traceable(prompt: Union[str, StringPromptValue], config: dict = default_config): # config: dict = generation_config):
    if type(prompt) is StringPromptValue:
        prompt = prompt.to_string() # Umwandeln in einen str
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    response = model.generate(
        **inputs,
        max_new_tokens=config.get("max_new_tokens", 5000),
        do_sample=config.get("do_sample", True),
        temperature=config.get("temperature", 0.6),
        top_p=config.get("top_p", 0.95),
        top_k=config.get("top_k", 30),
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response['generated_text'] = tokenizer.decode(response.sequences[0], skip_special_tokens=False)
    return response

def format_docs(docs):
    # Sortieren der Dokumente
    # Gruppieren der Dokumente nach Quelle
    docs_by_source = {}
    is_tuple = isinstance(docs[0], tuple)
    for i, doc_item in enumerate(docs):
        doc = doc_item[0] if is_tuple else doc_item
        source = doc.metadata['source']
        if source not in docs_by_source: # Prüfen ob Quelle bereits in dict vorhanden ist
            docs_by_source[source] = [] # Aufnehmen neuer Quelle
        # Speichern des originalen Dokuments zusammen mit dem ursprünglichen Index
        docs_by_source[source].append({'item': doc_item, 'original_index': i})
    # Sortieren der Dokumente innerhalb jeder Quelle
    for source in docs_by_source:
        # Sortieren nach Seite und Startindex
        docs_by_source[source].sort(key=lambda x: (
            (x['item'][0] if is_tuple else x['item']).metadata.get('page', 0),
            (x['item'][0] if is_tuple else x['item']).metadata.get('start_index', 0)
        ))

    # Rekonstruieren der sortierten Dokumentenliste basierend auf der ursprünglichen Reihenfolge der Quellen
    sorted_docs = [None] * len(docs)
    processed_indices = set()

    for i, doc_item in enumerate(docs): # Iteration über die ursprüngliche Liste
        if i in processed_indices: # Überspringen bereits verarbeiteter Dokumente
            continue

        doc = doc_item[0] if is_tuple else doc_item # Auslesen des Dokuments
        source = doc.metadata['source'] # Auslesen der Quelle
        
        # Chronologisch sortierte Gruppe von Dokumenten der aktuellen Quelle
        sorted_group = docs_by_source[source]
        
        # Aufnehmen der sortierten Dokumente an den ursprünglichen Positionen
        for group_item in sorted_group:
            # Finden der nächsten freien Position in sorted_docs
            insert_pos = i
            while sorted_docs[insert_pos] is not None: # Sicherstellen, dass die Position frei ist
                insert_pos += 1
            sorted_docs[insert_pos] = group_item['item'] # Aufnehmen des Dokuments an der freien Position
            processed_indices.add(group_item['original_index']) # Markieren des Dokuments als verarbeitet
    
    # Filtern von None-Werten, falls vorhanden
    sorted_docs = [d for d in sorted_docs if d is not None]
    
    docs_formatted = []
    for doc_item in sorted_docs:
        doc = doc_item[0] if is_tuple else doc_item
        # Erstellen von dem Präfix zur Kontextualisierung der abgerufenen Dokumente
        if '\\' in doc.metadata['source']: # Prüfen ob Windows oder Linux Pfadstruktur vorliegt
            split_char = '\\'
        elif '/' in doc.metadata['source']:
            split_char = '/'
        else: # Zutreffend für bspw. Webseiten
            split_char = 'https://'

        source_name = doc.metadata['source'].split(split_char)[-1] # Sammlung von Informationen aus dem Dokument zur Kontextualisierung des Auszugs
        if split_char != 'https://': # Falls es sich um eine Datei handelt, wird die Dateiendung aus dem Namen entfernt
            source_name = source_name.split('.')[0] # Entfernen der Dateiendung falls es sich um eine Datei handelt
        prefix = f"Document: {source_name}"
        # Hinzufügen zusätzlicher Dokumentinformationen
        if 'title' in doc.metadata:
            prefix += f", Title: {doc.metadata['title']}"
        if 'page' in doc.metadata:
            prefix += f", Page: {doc.metadata['page']}"
        if 'author' in doc.metadata:
            prefix += f", Author: {doc.metadata['author']}"
        # Zusammenführen Präfix und Dokumenteninhalt
        doc_formatted = f'{prefix}\n{doc.page_content}'
        docs_formatted.append(doc_formatted)
    return "\n\n".join(docs_formatted)
    
#######################################################################################################################################################################################################
#Definition von default Größen bei der Kontextualisierung von Dokumenten
#######################################################################################################################################################################################################

#Definition des Sprachmodells # Verwendung von Pipeline Manager um Speicher zu sparen
config_context = {
    "do_sample": True, 
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": CHUNK_CONTEXT_LEN, #maximale Anzahl zu generierender Tokens gemäß globaler Variable CHUNK_CONTEXT_LEN
    "penalty_alpha":0,
    "top_k": 10,
}
llm_context_pipe_manager = PipelineManager(model, tokenizer, initial_params=config_context)
llm_context = llm_context_pipe_manager.get_pipeline()

def contextualize_chunks(doc_splits: list, doc_pages: list, prompt: PromptTemplate = prompt_contextualize, llm = llm_context):

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
            context_chunk = context_chunk.split(f"{parsing_string}\n")[-1] #parsen des LLM Outputs
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
    return df_splits.to_csv(directory, escapechar='\\') #abspeichern der Datei in ein auslesbares Format mittels load_csv_data

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
    Args:
    directory (str): Path to directory containing files to be embedded into a vector store.
    file_paths (str or list): Path(s) to specific files to be embedded into a vector store.
    data_path (str): Path to a (potential) vector store directory, where the file path omits the file extension .csv due to programmatic reasons. Used to embed exisitng documents with a different embedding model.
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

    # Frühes Laden der Vektordatenbank, um frühe Überprüfung durchzuführen, ob hochgeladene Dokumente bereits im Speicher vorhanden sind
    vector_store = vector_store_type.load_local(folder_path=vector_store_path,
                                    embeddings=embedding,
                                    allow_dangerous_deserialization=True)

    # Ausgabe aller Quellen, welche im Vektorspeicher hinterlegt sind
    vector_store_sources = [] # leere Liste, welche mit Quellen aus der Vektordatenbank befüllt wird
    for key in vector_store.docstore.__dict__['_dict']: # Ausgabe der keys des Vektorspeichers
        source = vector_store.docstore.__dict__['_dict'][key].metadata['source'] # Speichern des Quellpfades in variable
        source_name = source.rsplit('/')[-1] if '/' in source else source.rsplit('\\')[-1] # Extrahieren des Dateinamens aus Dateipfad # Dateipfad wird gemäß Linux oder Windows Pfadschreibweise getrennt
        #Sicherstellen, dass nur einzigartige Namen in Liste aufgenommen werden
        if source_name not in vector_store_sources: # Aufnahme des Dateinamens in Liste, wenn dieser noch nicht enthalten ist
            vector_store_sources.append(source_name)

    sources_list = os.listdir(directory) # Liste aller Quellen, welche in Vektordatenspeicher aufgenommen werden sollen
    for source in sources_list: # Iterieren über alle hochgeladenen Dateien

        if source in vector_store_sources: #Prüfung, ob Datei bereits in vektordatenbank vorhanden ist

            os.remove(FILE_PATH + '/' + source) # Löschen der Datei falls sie bereits im Vektorspeicher vorhanden ist
            print(f"Document {source} removed, because it is already saved in the vector store.")

    if len(os.listdir(directory)) > 0: #Überprüfen, ob Dateien im Ordner vorhanden sind bevor der Programmablauf zu Aufnahme neuer Dokumente gestartet wird
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

        # Hinzufügen von neuen Dokumenten zur Vektordatenbank
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
            df.to_csv(dir_df, escapechar='\\') #abspeichern der Datei in ein auslesbares Format mittels load_csv_data #Änderung bedingt durch Ergänzung des vorhandenen Dataframes
    else:
        print("No new documents found in the directory. No documents added to the vector store.")

    return vector_store, docs

##################################################################################################################################################################################################
#Textgenerierungsfunktionen
##################################################################################################################################################################################################

@chain
def query_parser(query: str):
    """Parses the output of the query analyzer, so it only returns the search_query"""
    new_query = query.split(sep="SEARCH: ", maxsplit=-1)[-1]
    task_description = "Given a web search query, retrieve relevant passages that are relevant to the query."
    task = f'Instruct: {task_description}\nQuery: {new_query}' #Ausgabe für Instruction trainierte Embedding Modelle #new_query fungiert als geeigneter Suchbegriff und task_description beschreibt den Suchkontext
    # Struktur der Task ist modellspezifisch anzupassen
    return task #new_query

# Definition der zu verwendenden Pipelines für llm und llm_query
config_text_gen = {
    "do_sample": True, 
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": 1000,
    "penalty_alpha":0.3,
    "top_k": 20,
    "streamer": streamer,
}
llm_pipe_manager = PipelineManager(model, tokenizer, initial_params=config_text_gen)
llm = llm_pipe_manager.get_pipeline()

config_rag_query = {
    "do_sample": False, 
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 100,
    "top_k": None,
    "streamer": None,
}
llm_query_pipe_manager = PipelineManager(model, tokenizer, initial_params=config_rag_query)
llm_query = llm_query_pipe_manager.get_pipeline()

def rag_gen(instruction : Union[str, List[str]], prompt: ChatPromptTemplate, retriever : VectorStoreRetriever, chat_history, extra = dict(), query_analysis = True,
            pipeline_textgen = llm, pipeline_rag_query = llm_query, print_sources = True,):

    #Überprüfen der Eingaben
    assert isinstance(instruction, str), "Please enter the instruction in form of a string"

    assert isinstance(extra, dict), "Please enter the extra variables in form of string(s) inside a dictionary, with the placeholder variable inside the prompt as keys and the corresponding strings as values"

    if isinstance(instruction, list): # Deaktivieren des Streamers falls eine Batcheingabe erfolgt
        config_text_gen = {
            "streamer": None
        }
        pipeline_textgen = llm_pipe_manager.get_pipeline(**config_text_gen) # Anpassen des Pipelinemanagers des Modells gemäß Nutzerwünschen


    rag_chain = (
        #{"context": retriever | format_docs, "input": RunnablePassthrough()} 
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))# Formatieren des eingegebenen Kontexts
        #format_docs formatiert die Docs, sodass nur konkatenierter page_content wiedergegeben wird und beschränkt die Textgenerierung
        | prompt
        | pipeline_textgen
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
        
        query_analyzer = {"input": RunnablePassthrough()} | prompt_query | pipeline_rag_query | query_parser | retriever #Definition der neuen retrieval chain, welche Inhalte gemäß der neuen Query heraussucht

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
        if isinstance(instruction, str): # Überprüfung ob Batcheingaben vorliegen oder nur einzelne Eingaben
            gen = rag_chain_with_source.invoke(rag_chain_dict)
        else:
            gen = rag_chain_with_source.batch(rag_chain_dict)
    else:
        if isinstance(instruction, str): # Überprüfung ob Batcheingaben vorliegen oder nur einzelne Eingaben
            gen = rag_chain_with_source.invoke(instruction) 
        else:
            gen = rag_chain_with_source.batch(rag_chain_dict)
    #Ablauf: Zunächst wird die Frage "question" für die Variable {input} in den Prompt eingegeben
    #Daraufhin lädt der Retriever Dokumente aus dem Vector Store der Variable vectorstore mit der höchsten Kosinusähnlichkeit (default)
    #Auf die beschaffenen Dokumente wird die Funktion format_docs() angewandt -> Die Dokumente werden konkateniert
    #Diese Dokumente werden in die Variable {context} von dem Prompt geladen
    #Falls zusätzliche Variablen in dem verwendeten Prompt vorliegen werden diese aus der Variable 'extra' herausgelesen
    #Die Variable prompt stellt den Kontext und das Eingabeschema bereit mit den 
    #Daraufhin wird für den gegebenen Kontext die Antwort des Sprachmodell über die Pipeline generiert
    #Zuletzt wird durch StrOutputParser das beste Ergebnis aus den generierten Tokens bezogen

    #Parsing der Ausgabe

    if isinstance(instruction, str): # Überprüfung ob Batcheingaben vorliegen oder nur einzelne Eingaben
        response = gen["answer"].split(f"{parsing_string_user}\n") #aufteilen string sodass nur Modellausgabe ohne Kontext stattfindet aber mit Frage #geeigneter Split ist promptabhängig
        response = response[-1] #Definition der Antwort als das letzte Listenelement, welches den String nach dem Separator "</context>\n\n" darstellt
        response_question = response.split(parsing_chat_turn_token)[0] #Speichern des Question Strings, welcher nach dem Token <|eot_id|> endet
        response_answer = response.split(f"{parsing_string}\n")[-1] #Trennen nach Prompt Tokens, welche den Antwortbereich für das Modell signalisieren
        response = response_question + response_answer #Zusammenfügen der Frage und Antwort
    else:
        for i in range(len(gen)):
            response = gen[i]["answer"].split(f"{parsing_string_user}\n") #aufteilen string sodass nur Modellausgabe ohne Kontext stattfindet aber mit Frage #geeigneter Split ist promptabhängig
            response = response[-1] #Definition der Antwort als das letzte Listenelement, welches den String nach dem Separator "</context>\n\n" darstellt
            response_question = response.split(parsing_chat_turn_token)[0] #Speichern des Question Strings, welcher nach dem Token <|eot_id|> endet
            response_answer = response.split(f"{parsing_string}\n")[-1] #Trennen nach Prompt Tokens, welche den Antwortbereich für das Modell signalisieren
            response = response_question + '\n' + response_answer #Zusammenfügen der Frage und Antwort
            print(response[i])

    if print_sources == True: #Fals die Quellenangaben mit ausgegeben werden sollen

        if isinstance(instruction, str): # Überprüfung ob Batcheingaben vorliegen oder nur einzelne Eingaben
            print("Sources:\n")
            for source in gen['context']:
                if isinstance(source, tuple): #Fallunterscheidung falls retrieval_scores ausgegeben werden -> dies resultiert in Ausgabe von tuple für source mit source[1] als retrieval score
                    print(source[0].metadata)
                    print(f"Retrieval Score: {source[1]}")
                else:
                    print(source.metadata)
        else:
            for i in range(len(gen)):
                for source in gen[i]['context']:
                    if isinstance(source, tuple): #Fallunterscheidung falls retrieval_scores ausgegeben werden -> dies resultiert in Ausgabe von tuple für source mit source[1] als retrieval score
                        print(f"Sources for the input '{gen[i]['input']}':\n")
                        print(source[0].metadata)
                        print(f"Retrieval Score: {source[1]}")
                    else:
                        print(f"Sources for the input '{gen[i]['input']}':\n")
                        print(source.metadata)

    # Aufräumen
    gc.collect() #Befreien der GPU Memory
    torch.cuda.empty_cache() #Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi

    return gen, response_answer #Ausgabe der response answer zur Weitergabe an die Chat History