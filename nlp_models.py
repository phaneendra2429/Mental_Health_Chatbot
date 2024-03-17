import torch
from transformers import DistilBertForSequenceClassification
import os
# # Get the directory path of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))
# model = DistilBertForSequenceClassification.from_pretrained("model.safetensors")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lxs1/DistilBertForSequenceClassification_6h_768dim")
model = AutoModelForSequenceClassification.from_pretrained("lxs1/DistilBertForSequenceClassification_6h_768dim")


# from transformers import DistilBertTokenizerFast
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def sentiment_class(summarized_text):    
    '''
    # 1 = non-depressed
    # 0 = depressed
    returns: example:- array([[0.00493283, 0.9950671 ]], dtype=float32)
    '''
    inputs = tokenizer(summarized_text, padding = True, truncation = True, return_tensors='pt').to('cuda')
    outputs = model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    return predictions

def pattern_classification():
    return result

def corelation_analysis():
    return result