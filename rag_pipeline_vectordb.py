from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import chromadb
from chromadb.utils import embedding_functions
import os

# Reference : https://towardsdatascience.com/rag-how-to-talk-to-your-data-eaf5469b83b0


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory="Data/chroma"
chroma_client = chromadb.PersistentClient(path=persist_directory)


# https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

def get_file_paths_recursively(folder_path):
    file_paths = []
    for root, directories, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def vdb_csv_loader(file_paths):
    for i in range(len(file_paths)):
        loader = CSVLoader(file_path=file_paths[i], encoding="latin-1")
        db = Chroma.from_documents(documents=loader.load(), embedding=embedding_function, collection_name= "mental_health_csv_collection", persist_directory=persist_directory) # pars to imclude (docs, emb_fun, col_name, direct_path)

###
def generate_csv_vector_db() -> None:
    
     # Get the directory path of the current script
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #folder_path = os.path.join(script_dir, 'Data/csv') 
    folder_path = "Data/csv"
    file_paths = get_file_paths_recursively(folder_path)

    #loaded all the files
    vdb_csv_loader(file_paths)

###
pdf_collection = Chroma(collection_name="mental_health_pdf_collection", embedding_function=embedding_function, persist_directory=persist_directory)      
def vdb_pdf_loader(file_paths):
    for i in range(len(file_paths)):
        loader = PyMuPDFLoader(file_path=file_paths[i])
        documents  = loader.load()
    
        store = InMemoryStore()
        rag_retriever = ParentDocumentRetriever(
            vectorstore=pdf_collection,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        rag_retriever.add_documents(documents)


def generate_pdf_vector_db() -> None:
    
    # Get the directory path of the current script
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #folder_path = os.path.join(script_dir, '/Data/pdf')
    folder_path = "Data/pdf"
    file_paths = get_file_paths_recursively(folder_path)
    vdb_pdf_loader(file_paths)


def vectordb_load():     
    # call csv loader
    #generate_csv_vector_db()

    # call PDF loader
    generate_pdf_vector_db()

# call vector db load
vectordb_load()

