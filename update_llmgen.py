## All Imports
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
import os
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


#preparation Data
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.2, token=HUGGINGFACEHUB_API_TOKEN
)
ch = ChatMessageHistory()
memory = ConversationBufferMemory(
    llm=llm, 
    memory_key="chat_history",
    return_messages=True,
    output_key='answer',
    input_key='question')
condense_question = """Given the following conversation and a follow-up message,
rephrase the follow-up message to a stand-alone question or instruction that
represents the user's intent precisely, add context needed if necessary to generate a complete and
unambiguous question, only based on the on the Follow up Question and chat history, don't make up messages.
Maintain the same question intent as the follow up input message.\n
Chat History:
{chat_history}\n
Follow Up Input: {question}
Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(condense_question)
condense_question_prompt.pretty_print()

system_message_template = """You're a Mental Health Specialist. Support those with Depressive Disorder.
Listen compassionately, respond helpfully. For casual talk, be friendly. For facts, use context.
If unsure, say, 'Out of my knowledge.' Always stay direct.
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
----------------
{context}"""

human_message_template = """User Query: {question} Answer:"""

messages = [
SystemMessagePromptTemplate.from_template(system_message_template),
#HumanMessagePromptTemplate.from_template("{question}")
HumanMessagePromptTemplate.from_template(human_message_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)
qa_prompt.pretty_print()
vectorstore = Chroma(collection_name="test_collection", persist_directory="Data/chroma", embedding_function=embedding_function )
retriever = vectorstore.as_retriever()

# This controls how the standalone question is generated.
# Should take `chat_history` and `question` as input variables.
# template = (
#     "Combine the chat history and follow up question into "
#     "a standalone question. Chat History: {chat_history}"
#     "Follow up question: {question}"
# )
# prompt = PromptTemplate.from_template(template)
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
# chain = ConversationalRetrievalChain(
# #     combine_docs_chain=combine_docs_chain,
#     memory= memory,
#     retriever=retriever,
#     question_generator=llm_chain
# )
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=retriever,
    memory = memory,
    return_source_documents=False,
    verbose=True,
    condense_question_prompt=condense_question_prompt,
    # chain_type = "stuff",
    combine_docs_chain_kwargs={'prompt': qa_prompt} # https://github.com/langchain-ai/langchain/issues/6879
)

history=ChatMessageHistory()
def llm_generation(question: str):
    llm_answer = retrieval_chain.invoke({'question':question, 'chat_history':history.messages})['answer'] #Answer = Dict Key = Latest response by the AI
    history.add_user_message(question)
    history.add_ai_message(llm_answer)
    return llm_answer
