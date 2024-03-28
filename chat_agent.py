import streamlit as st
from dotenv import load_dotenv, find_dotenv
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


# Please ensure you have a .env file available with 'HUGGINGFACEHUB_API_TOKEN'
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN  = os.environ["HUGGINGFACEHUB_API_TOKEN"]

repo_id="mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=512, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
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



# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
# memory.save_context({"input": "hi"}, {"output": "whats up"})

# def ConvoLLM(query: str): 
#     conversation.predict(input=query)

#---------------------------------------------------------------------------------------------------------------------------------------
# print(conversation.predict(input="I am feeling low"))
# print(conversation.predict(input="I am alone at home"))
# print(conversation.memory.buffer)
query2 = " "
template = """ Act as an expert mental health therapist, and conduct therapy sessions with the user. You are an expert Mental Health therapist who is asking the user questions to learn what professional mental health well-being advice could help the user.
    Your goal is to analyse their mental health problem, based following input:{input}. You will always ask questions to the user to get them to explain more about whatever mental health condition is ailing them.
    DO NOT give the user any mental health advice or medical advice, ONLY ask for more information about their symptoms.
    Do not show your thought process, only output a single question.  Your output should contain consolation related to the query and a single question.
    Only ask one question a time. 
    
    Current conversation:
    {history}
    
    Human: {input}
    AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history","input"], template=template)
memory = ConversationBufferMemory(llm=llm)
    # memory.save_context({"input": "hi"}, {"output": "whats up"})
    # memory.save_context({"input": "not much you"}, {"output": "not much"})
    # memory.save_context({"input": "feeling sad"}, {"output": "I am happy you feel that way"})

conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        memory=memory,
        verbose=True
    )

def convo(query):
    global conversation
    
    response = conversation.predict(input=query)
    # memory.save_context({"input": query}, {"output": ""})
    global query2
    query2 = query2 + "," + query
    print("\n chat_agent.py----------",memory.chat_memory)
    summary = query2
    return response, summary




