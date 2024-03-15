from langchain_community.llms import HuggingFaceEndpoint
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
import streamlit as st

HUGGINGFACEHUB_API_TOKEN = "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"


os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


def identify_emotion(text):
  # Use a sentiment analysis library here (e.g., from transformers)
  # sentiment_pipeline = pipeline("sentiment-analysis", model="mistralai/Mistral-7B-Instruct-v0.2")
  sentiment_pipeline = pipeline('sentiment-analysis')
  emotion = sentiment_pipeline(text)[0]['label']
  print(emotion)
  return emotion


template = """I hear you're feeling {emotion}. Can you tell me more about what's going on?"""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.2, token=HUGGINGFACEHUB_API_TOKEN
)
llm_chain = LLMChain(prompt=prompt, llm=llm)


# App title
st.set_page_config(page_title=" Llama 2 Chatbot")

st.title("Mistral Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    emotion = identify_emotion(prompt)  # Analyze user input for emotion
    template = f"""I hear you're feeling {emotion}. Can you tell me more about what's going on?"""

    response = llm_chain.run(template)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Additional Resources (Consider adding based on user's situation)
    if any(word in response.lower() for word in ["crisis", "hopeless", "self-harm"]):
        st.markdown("**If you are in crisis, please reach out for help.** Here are some resources:")
        st.markdown("- National Suicide Prevention Lifeline: 988")
        st.markdown("- Crisis Text Line: Text HOME to 741741")
