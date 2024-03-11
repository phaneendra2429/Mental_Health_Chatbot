import streamlit as st
from llm_generator import llm_generator

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

st.title('Therapist')
# user_prompt = st.text_input('Say Hello to start your conversation')

with st.chat_message('user'):
    user_prompt = st.text_input('Say Hello to start your conversation')

if st.button('Response') and user_prompt:
    with st.spinner("I'm thinking..."):
        human_responses, ai_responses = llm_generator(user_prompt)
        with st.chat_message("assistant"):
            st.write(ai_responses[-1])