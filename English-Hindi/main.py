import pandas as pd
import numpy as np

import os
import string
from string import digits
import re

df = pd.read_csv('./data/Hindi_English_Truncated_Corpus.csv')
df = df.sample(n=25000, random_state=42)
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


df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: "START_" + x + "_END")

print(df.head())

# Vocabulary


# data split


# data modeling