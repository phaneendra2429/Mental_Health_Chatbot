# Docs:- https://huggingface.co/meta-llama/LlamaGuard-7b
from dotenv import load_dotenv, find_dotenv
import os
import requests
model_id = "meta-llama/LlamaGuard-7b"


# Please ensure you have a .env file available with 'HUGGINGFACEHUB_API_TOKEN'
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN  = os.environ["HUGGINGFACEHUB_API_TOKEN"]

#updated on March 24th

#reading unsafe categories file
try:
   with open('./prompts/llama_guard-unsafe_categories.txt', 'r') as file:
        unsafe_categories = file.read()
except FileNotFoundError:
    print("File not found. Please check the file path.")
    unsafe_categories = ""  # Or handle the missing file as appropriate for your application
except PermissionError:
    print("Permission denied. You don't have the rights to read the file.")
    unsafe_categories = ""  # Or handle the lack of permissions as appropriate
except Exception as e:  # Catch any other exceptions
    print(f"An error occurred while reading the file: {e}")
    unsafe_categories = ""  # Or handle unexpected errors as appropriate

#added on March 24th
def moderation_prompt_for_chat(chat):
  prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.
  <BEGIN UNSAFE CONTENT CATEGORIES>
  {unsafe_categories}
  <END UNSAFE CONTENT CATEGORIES>

  <BEGIN CONVERSATION>

  User : {chat}

  <END CONVERSATION>

  Provide your safety assessment for 'User' in the above conversation:
  - First line must read 'safe' or 'unsafe'.
  - If unsafe, a second line must include a comma-separated list of violated categories.[/INST]"""
  return prompt

def query(payload):
    API_URL = "https://okoknht2arqo574k.us-east-1.aws.endpoints.huggingface.cloud"
    bearer_txt = f'Bearer {HUGGINGFACEHUB_API_TOKEN}'
    headers = {
        "Accept": "application/json",
        "Authorization": bearer_txt,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # This will raise an exception for HTTP error responses
        return response.json(), None
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred: {http_err}"
        print(error_message)
    except requests.exceptions.ConnectionError:
        error_message = "Could not connect to the API endpoint."
        print(error_message)
    except Exception as err:
        error_message = f"An error occurred: {err}"
        print(error_message)
    
    return None, error_message


def moderate_chat(chat):
    prompt = moderation_prompt_for_chat(chat)
    	
    output, error_msg = query({
        "inputs": prompt,
        "parameters": {
		"top_k": 1,
		"top_p": 0.2,
		"temperature": 0.1,
		"max_new_tokens": 512
	}
    })
  
    return output, error_msg


#added on March 24th
def load_category_names_from_string(file_content):
    """Load category codes and names from a string into a dictionary."""
    category_names = {}
    lines = file_content.split('\n')
    for line in lines:
        if line.startswith("O"):
            parts = line.split(':')
            if len(parts) == 2:
                code = parts[0].strip()
                name = parts[1].strip()
                category_names[code] = name
    return category_names

def get_category_name(input_str):
    """Return the category name given a category code from an input string."""
    # Load the category names from the file content
    category_names = load_category_names_from_string(unsafe_categories)
    
    # Extract the category code from the input string
    category_code = input_str.split('\n')[1].strip()
    
    # Find the full category name using the code
    category_name = category_names.get(category_code, "Unknown Category")
    
    #return f"{category_code} : {category_name}"
    return f"{category_name}"


