from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Import CSV Files to the VectorDB
# Reference : https://towardsdatascience.com/rag-how-to-talk-to-your-data-eaf5469b83b0

# df_mental_health = pd.read_excel("/content/drive/MyDrive/Team 5/Depression_dataset_preprocessed (1).xlsx", sheet_name= "98_row_Mental_Health_FAQs")
# df_counsellor_chats = pd.read_excel("/content/drive/MyDrive/Team 5/Depression_dataset_preprocessed (1).xlsx", sheet_name= "Counsellor_Chats")
# df_human_therapist = pd.read_excel("/content/drive/MyDrive/Team 5/Depression_dataset_preprocessed (1).xlsx", sheet_name= "99_rows_Human_&_Therapist")

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

loader = PyMuPDFLoader(os.path.join(script_dir, 'Data','pdf', 'Depression Help Guide.pdf'))
documents  = loader.load()

# create the open-source embedding function
# Docs:- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

persist_directory="Data/chroma"
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embedding_function, persist_directory=persist_directory)

# The storage layer for the parent documents
store = InMemoryStore()

def instantiate_rag():
    rag_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    rag_retriever.add_documents(documents)
    return rag_retriever


#call the above method
instantiate_rag()