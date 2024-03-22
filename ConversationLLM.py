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

# template="""Act as a therapist, and conduct therapy sessions with the user. Your goal analyse their mental health 
# problem, based following input:{query}. Do not show your thought process, only output a single question. 
# Your output should contain consolation related to the query and a single question. Only ask one question at a time."""

# def ConvoLLM(query: str): 
#     prompt_template=PromptTemplate(input_variables=['query'],template= template)
#     prompt_template.format(query= query)
#     chain=LLMChain(llm=llm,prompt=prompt_template)
#     response = chain.run(query) 
#     return response

#---------------------------------------------------------------------------------------------------------------------------------------


template = """Act as a therapist, and conduct therapy sessions with the user. Your goal analyse their mental health 
problem, based following input:{input}. Do not show your thought process, only output a single question. 
Your output should contain consolation related to the query and a single question. Only ask one question at a time.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# memory=ConversationBufferMemory()
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=llm)  
)

# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
# memory.save_context({"input": "hi"}, {"output": "whats up"})

# def ConvoLLM(query: str): 
#     conversation.predict(input=query)

#---------------------------------------------------------------------------------------------------------------------------------------


st.set_page_config(layout="wide")

# st.title('Mental Health Counseling Chatbot')

# Adjust sidebar width to take half the screen
col1, col2 = st.columns([2, 3])

# Function to update recommendations in col1
def update_recommendations(new_content):
    with col1:
        st.header("Recommendation")
        st.write(new_content.memory.load_memory_variables({}))  # Update the content with new_content



# Main content area
with col2:
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
            response = conversation.predict(input=user_prompt)
            # memory.save_context({"input": user_prompt}, {"output":response})
            # print(conversation.memory.load_memory_variables())
            time.sleep(0.2)
            st.write_stream(response_generator(response))
            # Update recommendations based on the assistant's response (replace with your logic)
            # new_recommendations = "Here are some updated recommendations based on the assistant's reply: ..."  # Modify this line
            update_recommendations(conversation)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
