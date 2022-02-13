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


# Create the Data and handling the problem of RuntimeError: CUDA error: device-side assert triggered

# f = open('Annotation_work', 'r')
# lines = f.readlines()
# for i in range(len(lines)):
#     lines[i] = lines[i].split(';')

# data = []
# for i in range(len(lines)):
#     lines[i][-1] = lines[i][-1].strip()
#     if lines[i][-1] == '-1':
#         data.append([lines[i][3], '1', '0', '0'])
#     elif lines[i][-1] == '0':
#         data.append([lines[i][3], '0', '1', '0'])
#     elif lines[i][-1] == '1':
#         data.append([lines[i][3], '0', '0', '1'])

# with io.open("Bert_Data.csv", mode='w', encoding='UTF8', newline='') as toWrite:
#     writer = csv.writer(toWrite)
#     writer.writerow(["Text", "-1", "0", "1"])
#     writer.writerows(data)


train_path = './Monument/Bert_Data.csv'
df = pd.read_csv(train_path)

target_list = ['-1', '0', '1']

# hyperparameters
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-05


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['Text']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }


train_size = 0.8
train_df = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
val_df = df.drop(train_df.index).reset_index(drop=True)

print(train_df.shape)
print(val_df.shape)

train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0
                                                )

val_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=VALID_BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=0
                                              )

print(len(train_data_loader))
print(len(val_data_loader))

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


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


model = BERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

val_targets = []
val_outputs = []


def train_model(n_epochs, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            #print('yyy epoch', batch_idx)
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            # if batch_idx%5000==0:
            #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('before loss data in training', loss.item(), train_loss)
            train_loss = train_loss + \
                ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            #print('after loss data in training', loss.item(), train_loss)

        print('############# Epoch {}: Training End     #############'.format(epoch))

        print('############# Epoch {}: Validation Start   #############'.format(epoch))
        ######################
        # validate the model #
        ######################

        model.eval()

        print(len(validation_loader))

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(
                    device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + \
                    ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(
                    outputs).cpu().detach().numpy().tolist())

            print(
                '############# Epoch {}: Validation End     #############'.format(epoch))
            # calculate average losses
            #print('before cal avg train loss', train_loss)
            train_loss = train_loss/len(training_loader)
            valid_loss = valid_loss/len(validation_loader)
            # print training/validation statistics
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                  epoch,
                  train_loss,
                  valid_loss
                  ))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            # TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

        print('############# Epoch {}  Done   #############\n'.format(epoch))

    return model


ckpt_path = "./Monument/curr_ckpt"
best_model_path = "./Monument/best_model.pt"

trained_model = train_model(EPOCHS, train_data_loader,
                            val_data_loader, model, optimizer, ckpt_path, best_model_path)
