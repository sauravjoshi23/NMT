#!/usr/bin/env python
"""

Neural SPARQL Machines - Dataset preparation.

'SPARQL as a Foreign Language' by Tommaso Soru and Edgard Marx et al., SEMANTiCS 2017
https://arxiv.org/abs/1708.07624

Version 2.0.0

"""
from fileinput import filename
import tensorflow as tf

import unicodedata
import re
import io
import numpy as np


def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿_]+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='',lower=False)
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def maxlength(data):
  maxlen = 0
  for x in data:
    maxlen = max(maxlen, len(x))
  
  return maxlen
  

def load_glove_embeddings(filename):
  embedding_dictionary = dict()
  with open(filename,'r') as glove_file:
      for line in glove_file:
          values=line.split()
          word=values[0]
          vectors=np.asarray(values[1:],'float32')
          embedding_dictionary[word]=vectors
  glove_file.close()

  return embedding_dictionary

def create_embedding_matrix(embedding_dictionary, vocab_size, MAXLEN, tokenizer):
  embedding_matrix = np.zeros((vocab_size, MAXLEN))
  for word, index in tokenizer.word_index.items():
      embedding_vector = embedding_dictionary.get(word)
      if embedding_vector is not None: 
          embedding_matrix[index] = embedding_vector
        
  return embedding_matrix


def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
  inp_lang_wp, targ_lang_wp  = create_dataset(path, num_examples)

  input_tensor, inp_lang = tokenize(inp_lang_wp)
  target_tensor, targ_lang = tokenize(targ_lang_wp)

  return input_tensor, target_tensor, inp_lang, targ_lang


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
