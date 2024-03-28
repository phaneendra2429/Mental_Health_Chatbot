from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import pickle
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate



# Please ensure you have a .env file available with 'HUGGINGFACEHUB_API_TOKEN'
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN  = os.environ["HUGGINGFACEHUB_API_TOKEN"]

repo_id="mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=512, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)



def recommend(query):
    with open('saved_object.pkl', 'rb') as f:
        obj = pickle.load(f)

    # Access the data attribute of the reconstructed object
    print(obj)

    vectorstore = Chroma(
        collection_name="split_parents", embedding_function=embedding_function, persist_directory="Data\chroma")
    rag = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=obj,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    pg=rag.get_relevant_documents(query)
    print("\n\n Retrival:",pg[0].page_content)

def recommend2(query):
    vectorstore = Chroma(
        collection_name="split_parents", embedding_function=embedding_function, persist_directory="Data\chroma")
    retriver = vectorstore.as_retriever()

    prompt="you are a mental health therapist, talking to a person with who is facing some mental health issues. Following is the user feeling {question}"
    prompt = PromptTemplate(input_variables=['question'],template=prompt)

    chain1 = LLMChain(llm=llm, prompt=prompt)
    doc_chain = load_qa_chain(llm, chain_type="map_reduce")

    chain = ConversationalRetrievalChain(
        retriever=retriver,
        question_generator=chain1,
        combine_docs_chain=doc_chain
    )
    chat_history = []
    # query = "I feel sad"
    result = chain({"question": query, "chat_history": chat_history})
    print("---------------\nSummary Bot:",query)
    print("this is the response from AI Bot:",result["answer"])

    return result["answer"]

# print(recommend2("i am feeling sad"))
