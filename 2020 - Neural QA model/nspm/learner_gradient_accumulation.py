#!/usr/bin/env python
"""

Neural SPARQL Machines - Learner module.

'SPARQL as a Foreign Language' by Tommaso Soru and Edgard Marx et al., SEMANTiCS 2017
https://arxiv.org/abs/1708.07624

Version 2.0.0

"""
import argparse
import tensorflow as tf
import numpy as np
import os
import io
import time
import json

import data_gen

from nmt import NeuralMT, NeuralMTConfig


@tf.function
def train_step(inp, targ, enc_hidden, targ_lang, batch_size, neural_mt):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = neural_mt.encoder(inp, enc_hidden)

        # Encoder hidden states to be passed to the decoder as input
        dec_hidden = enc_hidden

        # Every decoder input gets 2 input -> 1 is <start> and 2 is enc hidden states as above
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        # targ.shape[0] -> Total number of data in a batch i.e batch size
        # targ.shape[1] -> Length of a sequence i.e same for all data by padding
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = neural_mt.decoder(dec_input, dec_hidden, enc_output)

            loss += neural_mt.loss_function(targ[:, t], predictions)

            # using teacher forcing i.e we pass not what we get from the previous decoder cell but the target data itself
            # so here we select all the rows in batch but only single column t as the input i.e single word from each targ dataset
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = neural_mt.encoder.trainable_variables + neural_mt.decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    # neural_mt.optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, gradients

def gradient_sum(GRADS):
    '''sum of gradient which is accumulated by each mini batch calculation '''
    grads_sum = []
    for l_i in range(len(GRADS[0])):
        tmp = []
        for a_i in range(len(GRADS)):
            tmp.append(GRADS[a_i][l_i])
        grads_sum.append(tf.math.add_n(tmp))
    return grads_sum


def gradient_mean(SUM_GRADS, batch_accumulate_num):
    '''calucate mean gradient from sum gradient'''
    assert len(SUM_GRADS)==1
    mean_grads = []
    for l_i in range(len(SUM_GRADS[0])):
        mean_grads.append(SUM_GRADS[0][l_i] / batch_accumulate_num)
    return mean_grads


def learn(input_dir, output_dir, input_file, epochs, save_nmt=False):

    dataset, vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_size, batch_accumulate_num, example_input_batch, steps_per_epoch, targ_lang, max_length_targ, max_length_inp, inp_lang, embedding_en, embedding_sparql = data_gen.data_gen(input_dir, output_dir, input_file)

    config = NeuralMTConfig(vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_size, example_input_batch, max_length_targ, max_length_inp, inp_lang, targ_lang, embedding_en, embedding_sparql)
    neural_mt = NeuralMT(config)

    encoder, decoder, checkpoint = neural_mt.encoder, neural_mt.decoder, neural_mt.checkpoint

    given_dir = input_dir + '/training_log.txt'

    checkpoint_prefix = os.path.join(input_dir + '/training_checkpoints', "ckpt")

    train_l = []

    for epoch in range(epochs):
        empty_s = " "
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for batch in range(steps_per_epoch):

            # gradient accumulation
            GRADS = []
            train_iter = iter(dataset)
            for acc_i in range(batch_accumulate_num):
                inp, targ = next(train_iter)
                batch_loss, grads = train_step(inp, targ, enc_hidden, targ_lang, batch_size, neural_mt)
                if acc_i == 0:
                    GRADS = [grads]
                else:
                    GRADS.append(grads)
                    GRADS = [gradient_sum(GRADS)]
                
                total_loss += (batch_loss/batch_accumulate_num)

            # update with mean gradients
            GRADS = gradient_mean(GRADS, batch_accumulate_num)
            
            variables = neural_mt.encoder.trainable_variables + neural_mt.decoder.trainable_variables
            neural_mt.optimizer.apply_gradients(zip(GRADS, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f} \n'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        empty_s = 'Epoch {} Loss {:.4f} \n'.format(epoch + 1,
                                            total_loss / steps_per_epoch)
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        empty_s += 'Time taken for 1 epoch {} sec\n'.format(time.time() - start)
        train_l.append(empty_s)

    filelog = open(given_dir, 'w', encoding="utf8")
    filelog.writelines(train_l)
    filelog.close()

    if save_nmt:
        neural_mt.save(output_dir)

    return neural_mt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        '--input', dest='input', metavar='inputDirectory', help='dataset directory', required=True)
    requiredNamed.add_argument(
        '--output', dest='output', metavar='outputDirectory', help='dataset directory', required=True)
    requiredNamed.add_argument(
        '--input_file', dest='input_file', metavar='inputFileDirectory', help='dataset directory', required=True)
    requiredNamed.add_argument(
        '--epochs', dest='epochs', type=int, help='number of epochs', default=10)

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    input_file = args.input_file
    epochs = args.epochs

    neural_mt = learn(input_dir, output_dir, input_file, epochs, save_nmt=True)
