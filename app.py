import streamlit as st
import llm_generator
from llm_generator import llm_generation

import time

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

# Set the page to wide mode
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
        response = llm_generation(user_prompt)
        time.sleep(0.2)
        st.write_stream(response_generator(response))
        
    st.session_state.messages.append({"role": "assistant", "content": response})