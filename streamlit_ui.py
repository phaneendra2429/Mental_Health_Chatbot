import streamlit as st
st.set_page_config(layout='wide')
from llm_generator import llm_generation
from llama_guard import moderate_chat
import time

# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
st.title('Therapist')

# Set the page to wide mode


def response_generator(response):
    '''
    Responds the text in a stream-like way / ChatGPT way
    '''
    response_buffer = response
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
if user_prompt := st.chat_input("Please tell me about your mental health condition and we can explore together potential advice that could help you."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        chat = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""}]
        response = ""
        guard_status = moderate_chat(chat)
        if 'unsafe' in guard_status:
            response = 'Un safe input, Please repharse your question'
        else:
            response = llm_generation(user_prompt) # User input is passed to LLM as query - Would return a response based on ConversationRetrive
        st.write_stream(response_generator(response))
        time.sleep(1)
    st.session_state.messages.append({"role": "assistant", "content": response})