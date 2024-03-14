import streamlit as st
from llm_generator import llm_generation
import llm_generator

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

st.title('Therapist')

with st.chat_message('user'):
    user_prompt = st.text_input('Say Hello to start your conversation')

if st.button('Response') and user_prompt:
    with st.spinner("I'm thinking..."):
        llm_generation(user_prompt)
        with st.chat_message("assistant"):
            st.write(llm_generator.human_responses[-1], llm_generator.ai_responses[-1])