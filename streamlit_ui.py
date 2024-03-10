import streamlit as st
from llm_generator import LLM_generator

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

st.title('Therapist')
# user_prompt = st.text_input('Say Hello to start your conversation')

with st.chat_message('user'):
    user_prompt = st.text_input('Say Hello to start your conversation')

if st.button('Response') and user_prompt:
    with st.spinner("I'm thinking..."):
        output = LLM_generator(user_prompt)
        with st.chat_message("assistant"):
            # st.write("Hello human")
            st.write(output)
        with st.chat_message("user"):
            st.write("Hello human")
            #st.write(output)