import streamlit as st
import os
import time
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
) 
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import LLMChain, ConversationChain

HUGGINGFACEHUB_API_TOKEN ="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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id="mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.7, token=HUGGINGFACEHUB_API_TOKEN
)

def ConvoLLM(query: str): 
    prompt_template=PromptTemplate(input_variables=['query'],
    template="Act as a therapist, and conduct therapy sessions with the user. Your goal analyse their mental health problem, based following input:{query}.Do not show your thought process, only output a single question. Your output should contain consolation related to the query and a single question. Only ask one question at a time.")
    prompt_template.format(query= query)
    chain=LLMChain(llm=llm,prompt=prompt_template)
    response = chain.run(query) 
    return response

st.set_page_config(layout="wide")

st.title('Mental Health Counseling Chatbot')

def response_generator(response):
    '''
    responds the text with a type writter effect
    '''
    response_buffer = response.strip()
    for word in response_buffer.split():
        yield word + " "
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#with st.chat_message("assistant"):
    #st.write("Please tell me about your mental health condition and we can explore together. Potential mental health advice that could help you will be in the sidebar as we talk")

# Accept user input
#if user_prompt := st.chat_input("Hello, How are you doing today"):
if user_prompt := st.chat_input("Please tell me about your mental health condition and we can explore together potential advice that could help you."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = ConvoLLM(user_prompt)
        time.sleep(0.2)
        st.write_stream(response_generator(response))
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    