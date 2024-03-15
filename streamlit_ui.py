import streamlit as st
from llm_generator import llm_generation, update_list
import llm_generator

import random
import time

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

st.title('Therapist')

def response_generator():
    '''
    responds the text with a type writter effect
    '''
    response_buffer = llm_generator.ai_responses[-1]
    for word in response_buffer.split():
        yield word + " "
        time.sleep(0.1)

# Dont need this function here anymore
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
        llm_generation(user_prompt)
        time.sleep(2)
        update_list()
        time.sleep(2)
        response = st.write_stream(response_generator())
        #st.write(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})