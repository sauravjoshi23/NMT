#!/usr/bin/env python
"""

Neural SPARQL Machines - Neural Machine Translation.

'SPARQL as a Foreign Language' by Tommaso Soru and Edgard Marx et al., SEMANTiCS 2017
https://arxiv.org/abs/1708.07624

Version 2.0.0

"""
import tensorflow as tf

import pickle


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_en):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_en),
trainable=True)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_sparql):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_sparql),
trainable=True)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # x-> decoder input i.e <start>
    # hidden -> enc hidden states
    # enc_output -> for attention purposes
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    # We here compute the embedding of the token i.e <start> or hi, whatever we get from the decoder output cell
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    # concat embedding vector and the context vector
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


class NeuralMTConfig(object):
  """docstring for NeuralMTConfig"""
  def __init__(self, 
              vocab_inp_size, 
              vocab_tar_size, 
              embedding_dim, 
              units, 
              batch_size, 
              example_input_batch, 
              max_length_targ, 
              max_length_inp,
              inp_lang,
              targ_lang,
              embedding_en,
              embedding_sparql
              ):
    self.vocab_inp_size = vocab_inp_size
    self.vocab_tar_size = vocab_tar_size
    self.embedding_dim = embedding_dim
    self.units = units
    self.batch_size = batch_size
    self.example_input_batch = example_input_batch
    self.max_length_targ = max_length_targ
    self.max_length_inp = max_length_inp
    self.inp_lang = inp_lang
    self.targ_lang = targ_lang
    self.embedding_en = embedding_en
    self.embedding_sparql = embedding_sparql


class NeuralMT(object):

  """docstring for NeuralMT"""
  def __init__(self, config):
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                      reduction='none')

    encoder = Encoder(config.vocab_inp_size, config.embedding_dim, config.units, config.batch_size, config.embedding_en)

    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(config.example_input_batch, sample_hidden)
    print (f"Encoder output shape: (batch size, sequence length, units) {sample_output.shape}")
    print (f"Encoder Hidden state shape: (batch size, units) {sample_hidden.shape}")

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print(f"Attention result shape: (batch size, units) {attention_result.shape}")
    print(f"Attention weights shape: (batch_size, sequence_length, 1) {attention_weights.shape}")

    decoder = Decoder(config.vocab_tar_size, config.embedding_dim, config.units, config.batch_size, config.embedding_sparql)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((config.batch_size, 1)),
                                          sample_hidden, sample_output)

    print (f"Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}")

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    self.config = config
    self.encoder = encoder
    self.decoder = decoder
    self.optimizer = optimizer
    self.checkpoint = checkpoint             # TODO: find way to save the parameters needed to recreate these objects

  def loss_function(self, real, pred):
    # real -> Size = bs -> contains single true value of every sentence i.e column from 2d matrix [1, 23, 888, 0 , 0, 452]
    # pred -> Size = (bs, num of classes) -> It conains probabilities of every class
    # mask -> Size = bs -> If real = [1, 2, 3, 0], mask = [tr, tr, tr, false] because 0 was added for padding purposes and 
    # should not be included in calculating the loss
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # loss_ -> Size = bs -> calaculation of loss btw real and pred using SparseCategoricalCrossentropy
    loss_ = self.loss_object(real, pred)
    # casting i.e changing its type from bool to float32 -> [tr, fl, tr] = [1, 0, 1]
    mask = tf.cast(mask, dtype=loss_.dtype)
    # removing the padded tokens from loss [0.23, 0.34, 0.22] * [1, 0, 1] 
    loss_ *= mask
    # computes the mean of the loss array
    return tf.reduce_mean(loss_)

  def save(self, directory):
    with open(f"{directory}/neuralmt.pkl", "wb") as f_out:
      pickle.dump(self.config, f_out)

  @staticmethod
  def load(directory):
    with open(f"{directory}/neuralmt.pkl", "rb") as f:
      config = pickle.load(f)
    return config
    
