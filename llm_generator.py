import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory,ConversationSummaryBufferMemory

# LLM Generator
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Load environment variables from .env file
load_dotenv()
# Get the file path from the environment variable
project_folder_path = os.getenv('PROJECT_FOLDER_PATH')


config = {'max_new_tokens': 256,
          'temperature': 0.4,
          'repetition_penalty': 1.1,
          'context_length': 4096, # Set to max for Chat Summary, Llama-2 has a max context length of 4096
          }

llm = CTransformers(model=os.path.join(project_folder_path,'models/LLM/llama-2-7b-chat.Q2_K.gguf'),
                    callbacks=[StreamingStdOutCallbackHandler()],
                    config=config)


# Define system and user message templates
with open(os.path.join(project_folder_path,'prompts/system_message_template.txt'), 'r') as file:
    system_message_template = file.read().replace('\n', '')

with open(os.path.join(project_folder_path,'prompts/user_message_template.txt'), 'r') as file:
    user_message_template = file.read().replace('\n', '')

with open(os.path.join(project_folder_path,'prompts/condense_question_prompt.txt'), 'r') as file:
    condense_question_prompt = file.read().replace('\n', '')

# Create message templates
system_message = SystemMessagePromptTemplate.from_template(system_message_template)
user_message = HumanMessagePromptTemplate.from_template(user_message_template)

# Compile messages into a chat prompt template
messages = [system_message, user_message]
chatbot_prompt = ChatPromptTemplate.from_messages(messages)

# array_of_files
from rag_pipeline import instantiate_rag
retriever = instantiate_rag()

history = ChatMessageHistory()
# Provide the chat history when initializing the ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        input_key="question",
        llm=llm,
        max_token_limit=40,
        return_messages=True
    ),
    return_source_documents=False,
    chain_type="stuff",
    max_tokens_limit=100,
    condense_question_prompt= PromptTemplate.from_template(condense_question_prompt),
    combine_docs_chain_kwargs={'prompt': chatbot_prompt},
    verbose=True,
    return_generated_question=False,
)

def LLM_generator(question: str):
    answer = qa({"question": question,"chat_history":history.messages})["answer"]
    print("##------##")
    return answer

# Implement Classification

from nlp_models import sentiment_class, pattern_classification, corelation_analysis

#--------------------------------------------------------------------------------------------
#this variable is missing so I just hardcoded the sentiment
#is_depressed = sentiment_class(conversation_buffer)
#-----------------------------------------------------------------------------------------
is_depressed = [[0.5,0.5]]

if is_depressed[0][1] > 0.5:
    print('Not so depressed')
else: print('is_depressed')