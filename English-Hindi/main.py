import pandas as pd
import numpy as np

import os
import string
from string import digits
import re

df = pd.read_csv('./data/Hindi_English_Truncated_Corpus.csv')
df = df.sample(n=5000, random_state=42)
df['source'] = df['source'].astype('str')
df['english_sentence'] = df['english_sentence'].astype('str')
df['hindi_sentence'] = df['hindi_sentence'].astype('str')
print(df.head())

# data pre-processing

#lower
df['english_sentence'] = df['english_sentence'].apply(lambda x: x.lower())
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: x.lower())

# removing quotes
df['english_sentence'] = df['english_sentence'].apply(lambda x: re.sub("'", '', x))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

# removing punctuations
exclude = set(string.punctuation)
df['english_sentence'] = df['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# removing digits
hindi_digits = ['२','३','०','८','१','५','७','९','४','६']
df['english_sentence'] = df['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in digits))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in digits))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in hindi_digits))

# removing extra spaces
df['english_sentence'] = df['english_sentence'].apply(lambda x: x.strip())
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: x.strip())


df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: "START_ " + x + " _END")

print(df.head())

# Vocabulary
eng_words = set()
hin_words = set()

for x in df['english_sentence']:
    tokens = x.split()
    for tok in tokens:
        eng_words.add(tok)

for x in df['hindi_sentence']:
    tokens = x.split()
    for tok in tokens:
        hin_words.add(tok)

input_words = sorted(list(eng_words))
target_words = sorted(list(hin_words))


# data split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['english_sentence'], df['hindi_sentence'], test_size=0.2, random_state=42)



# data modeling

# https://www.kaggle.com/aiswaryaramachandran/english-to-hindi-neural-machine-translation