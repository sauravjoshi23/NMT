import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import csv
import numpy as np
from numpy.random import RandomState
from bert_utils import save_ckp, load_ckp

# hyperparameters
MAX_LEN = 256


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(
            'bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


def predict(device, text_pairs):

    train_path = './Monument/Bert_Data.csv'
    df = pd.read_csv(train_path)

    model = BERTClass()
    model = load_ckp('./Monument/curr_ckpt/content/curr_ckpt', model, '')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Copy the model to the GPU.
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    pred_labels = []
    for i in range(len(text_pairs)):
        example = text_pairs[i][1]
        print("Input Text: ", example)
        encodings = tokenizer.encode_plus(
            example,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        model.eval()
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(device, dtype=torch.long)
            attention_mask = encodings['attention_mask'].to(
                device, dtype=torch.long)
            token_type_ids = encodings['token_type_ids'].to(
                device, dtype=torch.long)
            output = model(input_ids, attention_mask, token_type_ids)
            final_output = torch.sigmoid(
                output).cpu().detach().numpy().tolist()
            # print(final_output)
            # print(df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
            pred_labels.append(int(df.columns[1:].to_list()[
                               int(np.argmax(final_output, axis=1))]))

    return pred_labels
