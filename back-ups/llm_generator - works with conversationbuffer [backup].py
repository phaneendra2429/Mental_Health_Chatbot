from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

# LLM Generator
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

config = {'max_new_tokens': 256,
          'temperature': 0.4,
          'repetition_penalty': 1.1,
          'context_length': 4096, # Set to max for Chat Summary, Llama-2 has a max context length of 4096
          }

llm = CTransformers(model='W:\\Projects\\LangChain\\models\\quantizedGGUF-theBloke\\llama-2-7b-chat.Q2_K.gguf',
                    callbacks=[StreamingStdOutCallbackHandler()],
                    config=config,
                    gpu_layers = 25)

# Implement another function to pass an array of PDFs / CSVs / Excels
from rag_pipeline import instantiate_rag
retriever = instantiate_rag()

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory

# Docs:- https://python.langchain.com/docs/integrations/chat/llama2_chat
from langchain_experimental.chat_models import Llama2Chat

# Define system and user message templates
with open('.\\prompts\\system_message_template.txt', 'r') as file:
            system_message_template = file.read().replace('\n', '')

template_messages = [
    SystemMessage(content=system_message_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

model = Llama2Chat(llm=llm)
# ConversationTokenBuffer
memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory, verbose=False)

# Decide wether to place this in streamlit.py
# or make a new post_process.py and import that to streamlit
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

human_inputs = ['Nothing logged yet']
ai_responses = ['Nothing logged yet']

def llm_generation(question):
    '''
    triggers the LLM    
    '''
    global human_inputs, ai_responses
    print(chain.invoke(input=question))
    human_inputs, ai_responses = extract_dialogues(memory.buffer_as_str)
    return memory.buffer_as_str

def is_depressed():
    ''''
    returns wether according to human inputs the person is depressed or not
    '''
    # Implement Classification
    all_user_inputs = ''.join(human_inputs)
    from nlp_models import sentiment_class, pattern_classification, corelation_analysis
    is_depressed = sentiment_class(all_user_inputs)
    return 'Not so depressed' if is_depressed[0][1] > 0.5 else 'is_depressed'