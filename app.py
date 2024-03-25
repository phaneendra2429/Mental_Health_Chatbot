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
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import LLMChain, ConversationChain
from chatbot import convo
from recommend import recommend2
from functools import cached_property

st.set_page_config(layout="wide")

# st.title('Mental Health Counseling Chatbot')

# Adjust sidebar width to take half the screen
col1, col2 = st.columns([2, 3])

# Function to update recommendations in col1
def update_recommendations(sum):
    # with col1:
    #     st.header("Recommendation")
    #     recommend = recommend2(sum)
    #     st.write(recommend)  # Update the content with new_content
    with st.sidebar:
        st.header("Recommendation")
        recommend = recommend2(sum)  # Assuming recommend2 doesn't require input
        st.write(recommend)
    
        # Add refresh button (simulated)
        if st.button("Refresh Chat"):
            st.rerun()

@cached_property
def get_recommendations():

    return "These are some updated recommendations."
# # Main content area
# with col2:
st.title('Mental Health Counseling Chatbot')
# ... (rest of your code for chatbot functionality) ...

def response_generator(response):
    '''
    responds the text with a type writter effect
    '''
    response_buffer = response.strip()
    for word in response_buffer.split():
        yield word + " "
        time.sleep(0.05)

with st.chat_message("assistant"):
    time.sleep(0.2)
    st.markdown("I am your Personal Therapist, How are you doing today?")  

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
        response,summary = convo(user_prompt)
        # print(conversation.memory.buffer)
        time.sleep(0.2)
        st.write_stream(response_generator(response))
        # print(conversation.memory.buffer)
        update_recommendations(summary)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
        
