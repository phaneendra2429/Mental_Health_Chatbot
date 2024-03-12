from dotenv import load_dotenv
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAI

load_dotenv()

CONVERSATION_BUFFER_MEMORY_MAX_TOKEN = 4000


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke({"input": query})

    return result, cb.total_tokens


def summarize_dialogue(input_dialogue):
    input_dialogue = input_dialogue.replace("Human:", "Patient:").replace("AI:", "Psychologist:")

    template = """You are an excellent document summarizer. Create a detailed summary of the dialogue between a 
    psychologist and a patient to aid other psychologists in crafting tailored therapeutic interventions. Summarize the 
    exchanges, highlighting the patient's psychological condition and therapeutic path, enabling fellow psychologists to 
    initiate effective healing processes.

    Dialogues: {dialogue}

    Please provide a summary of the dialogue:
    """

    summarize_prompt = PromptTemplate(
        input_variables=["dialogue"],
        template=template
    )

    summarizer = LLMChain(
        llm=llm,
        prompt=summarize_prompt
    )

    return summarizer.invoke({"dialogues": input_dialogue})


def create_new_memory(input_dialogue):
    summary = summarize_dialogue(input_dialogue)
    new_memory = ConversationBufferMemory()
    new_memory.chat_memory.add_user_message(HumanMessage(content="Can you give a summary based on my previous "
                                                                 "psychological therapy session?"))
    new_memory.chat_memory.add_ai_message(AIMessage(content=summary))

    return new_memory


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
conversation = ConversationChain(llm=llm,
                                 memory=ConversationBufferMemory())

while True:
    user_query = input("Input: ")

    if user_query is not None or user_query != "":
        response, tokens = count_tokens(conversation, user_query)
        print(f"Response: {response['response']}")

        if tokens >= CONVERSATION_BUFFER_MEMORY_MAX_TOKEN:
            dialogue = conversation.memory.buffer
            memory = create_new_memory(dialogue)
            conversation.memory = memory
    else:
        break
