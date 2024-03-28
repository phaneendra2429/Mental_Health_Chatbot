import streamlit as st
import llm_generator
from llm_generator import llm_generation
from llama_guard import moderate_chat, get_category_name
#added on March 24th
import time
from chat_agent import convo
from recommendation_agent import recommend2
from functools import cached_property


# ST : https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

# Set the page to wide mode
st.set_page_config(layout="wide")

# Set the title
st.title('Mental Health Counseling Chatbot')

# Adjust sidebar width to take half the screen
col1, col2 = st.columns([2, 3])

# Function to update recommendations in col1
def update_recommendations(sum):
    # with col1:
    #     st.header("Recommendation")
    #     recommend = recommend2(sum)
    #     st.write(recommend)  # Update the content with new_content
    with st.sidebar:
        st.header("Mental Health Advice")
        recommend = recommend2(sum)  # Assuming recommend2 doesn't require input
        st.write(recommend)
    
        # Add refresh button (simulated)
        if st.button("Refresh Chat"):
            st.rerun()

@cached_property
def get_recommendations():
    return "These are some updated recommendations."


def response_generator(response):
    '''
    responds the text with a type writter effect
    '''
    response_buffer = response.strip()
    for word in response_buffer.split():
        yield word + " "
        time.sleep(0.05)

with st.chat_message("assistant"):
    time.sleep(0.2)
    st.markdown("I am your Mental Health Counselar. How are you doing today?")  


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if 'llama_guard_enabled' is already in session state, otherwise initialize it
if 'llama_guard_enabled' not in st.session_state:
    st.session_state['llama_guard_enabled'] = True  # Default value to True

# Modify the checkbox call to include a unique key parameter
llama_guard_enabled = st.sidebar.checkbox("Enable LlamaGuard",
                                        value=st.session_state['llama_guard_enabled'],
                                            key="llama_guard_toggle")


# Update the session state based on the checkbox interaction
st.session_state['llama_guard_enabled'] = llama_guard_enabled

#with st.chat_message("assistant"):
    #st.write("Please tell me about your mental health condition and we can explore together. Potential mental health advice that could help you will be in the sidebar as we talk")

# Accept user input
#if user_prompt := st.chat_input("Hello, How are you doing today"):
if user_prompt := st.chat_input("Please tell me about your mental health condition and we can explore together potential advice that could help you that would be displayed on the left sidebar as we talk."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        print('llama guard enabled',st.session_state['llama_guard_enabled'])
        is_safe = True
        #added on March 24th
        unsafe_category_name = ""
        if st.session_state['llama_guard_enabled']:
            #guard_status = moderate_chat(user_prompt)
            guard_status, error = moderate_chat(user_prompt)
            if error:
             st.error(f"Failed to retrieve data from Llama Gaurd: {error}")
            else:
                if 'unsafe' in guard_status[0]['generated_text']:
                    is_safe = False
                    #added on March 24th
                    unsafe_category_name = get_category_name(guard_status[0]['generated_text'])
                    print(f'Guard status {guard_status}, Category name {unsafe_category_name}')
        if is_safe==False:
             #added on March 24th
            response = f"I see you are asking something about {unsafe_category_name} Due to eithical and safety reasons, I can't provide the help you need. Please reach out to someone who can, like a family member, friend, or therapist. In urgent situations, contact emergency services or a crisis hotline. Remember, asking for help is brave, and you're not alone."
        else:
            response,summary = convo(user_prompt)
            # print(conversation.memory.buffer)
            time.sleep(0.2)
            st.write_stream(response_generator(response))
            print("This is the response from app.py",response)
            update_recommendations(summary)
        
        st.session_state.messages.append({"role": "assistant", "content": response}) 

