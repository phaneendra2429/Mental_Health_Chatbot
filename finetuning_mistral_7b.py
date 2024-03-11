device = "cuda"

import datasets
import pandas as pd
from datasets import Dataset

df = pd.read_excel("TherapistHuman.xlsx")
df.head()

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, shuffle = True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_dict(test_df)
my_dataset_dict = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})

instruct_tune_dataset = my_dataset_dict
instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(80))
instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(20))

def create_prompt(sample):
  bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
  response = sample["HUMAN"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
  input = sample["ASSISTANT"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += "### Instruction:"
  full_prompt += "\n" + system_message
  full_prompt += "\n\n### Input:"
  full_prompt += "\n" + input
  full_prompt += "\n\n### Response:"
  full_prompt += "\n" + response
  full_prompt += eos_token

  return full_prompt

create_prompt(instruct_tune_dataset["train"][0])

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto', quantization_config=nf4_config, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded_output = tokenizer.batch_decode(generated_ids)
  return decoded_output[0].replace(prompt, "")

from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

from transformers import TrainingArguments

args = TrainingArguments(
  output_dir = "mistral_instruct_generation",
  #num_train_epochs=5,
  max_steps = 100, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 0.03,
  logging_steps=10,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='constant',
)

from trl import SFTTrainer
max_seq_length = 2048

trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt,
  args=args,
  train_dataset=instruct_tune_dataset["train"],
  eval_dataset=instruct_tune_dataset["test"]
)

trainer.train()
trainer.save_model("mistral_instruct_generation")

from huggingface_hub import notebook_login
notebook_login()

trainer.push_to_hub("ai-maker-space/mistral-instruct-generation")
merged_model = model.merge_and_unload()

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0]
