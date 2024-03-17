import streamlit as st
import llm_generator
from llm_generator import llm_generation

import time

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

st.title('Mental Health Therapist')

def response_generator(response):
    '''
    responds the text with a type writter effect
    '''
    response_buffer = response.strip()
    for word in response_buffer.split():
        yield word + " "
        time.sleep(0.1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_prompt := st.chat_input("Hello, How are you doing today"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = llm_generation(user_prompt)
        time.sleep(1)
        st.write_stream(response_generator(response))
        
    st.session_state.messages.append({"role": "assistant", "content": response})