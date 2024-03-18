from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
) # Docs:- https://python.langchain.com/docs/modules/model_io/prompts/message_prompts

#import chromadb

# LLM Generator
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory, ConversationBufferMemory
 
from langchain_experimental.chat_models import Llama2Chat
# Docs:- https://python.langchain.com/docs/integrations/chat/llama2_chat


HUGGINGFACEHUB_API_TOKEN =  "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" #HF_ACCESS_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Implement another function to pass an array of PDFs / CSVs / Excels
#from rag_pipeline import instantiate_rag
#retriever = instantiate_rag()

persist_directory="Data/chroma"
#chroma_client = chromadb.PersistentClient(persist_directory=persist_directory)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = Chroma(persist_directory = persist_directory, embedding_function = embedding_function, collection_name="split_parents")
retriever = vectors.as_retriever() #(k=6)


# Set the url to your Inference Endpoint below
#your_endpoint_url = "https://fayjubiy2xqn36z0.us-east-1.aws.endpoints.huggingface.cloud"

#how you can access HuggingFaceEndpoint integration of the free Serverless Endpoints API.
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    #endpoint_url=f"{your_endpoint_url}",
    repo_id=repo_id,
    #max_length=128,
    max_new_tokens=512,
    token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.1,
    repetition_penalty=1.1,
    #context_length: 4096, # Set to max for Chat Summary, Llama-2 has a max context length of 4096,
    stream=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    #top_k=10,
    #top_p=0.95,    
)


#model = Llama2Chat(llm=llm)
memory = ConversationBufferMemory(
    llm=llm, 
    memory_key="chat_history",
    return_messages=True,
    output_key='answer',
    input_key='question')


# Prompt Context Reference : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF , https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5#64b81e9b15ebeb44419a2b9e
# Reference:- https://github.com/langchain-ai/langchain/issues/5462

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

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=retriever,
    memory = memory,
    return_source_documents=False,
    verbose=True,
    condense_question_prompt=condense_question_prompt,
    # chain_type = "stuff",
    combine_docs_chain_kwargs={'prompt': qa_prompt}, # https://github.com/langchain-ai/langchain/issues/6879
)


human_inputs = ['Nothing logged yet']
ai_responses = ['Nothing logged yet']

history = ChatMessageHistory()

def llm_generation(question: str):
    llm_answer = retrieval_chain.invoke({'question':question, 'chat_history':history.messages})['answer'] #Answer = Dict Key = Latest response by the AI
    history.add_user_message(question)
    history.add_ai_message(llm_answer)
    return llm_answer



# Decide wether to place this in streamlit.py
# or make a new post_process.py and import that to streamlit
def extract_dialogues(text):
    '''
    returns a two lists for human and ai dialogues,
    '''
    human_dialogues = []
    ai_dialogues = []
    lines = text.split('\n')

    # Iterate through each line
    for line in lines:
        # Remove leading and trailing whitespace
        line = line.strip()

        # Check if the line starts with 'Human:' or 'AI:'
        if line.startswith('Human:'):
            # Extract the text after 'Human:'
            human_dialogues.append(line[len('Human:'):].strip())
        elif line.startswith('AI:'):
            # Extract the text after 'AI:'
            ai_dialogues.append(line[len('AI:'):].strip())
    return human_dialogues, ai_dialogues

def update_list():
    global human_inputs, ai_responses
    human_responses, ai_responses = extract_dialogues(memory.buffer_as_str)
    return 'responses updated'  


def is_depressed():
    ''''
    returns wether according to human inputs the person is depressed or not
    '''
    # Implement Classification
    all_user_inputs = ''.join(human_inputs)
    from nlp_models import sentiment_class, pattern_classification, corelation_analysis
    is_depressed = sentiment_class(all_user_inputs)
    return 'Not so depressed' if is_depressed[0][1] > 0.5 else 'is_depressed'