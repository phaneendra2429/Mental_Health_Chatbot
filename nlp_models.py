import torch
from transformers import DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained('distil-bert_finetuned_on_20001_depressed_data_set_[4_epoch, 16_batch size, 3e-5_LR]')

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

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

def pattern_classification:
    return result

def corelation_analysis:
    return result