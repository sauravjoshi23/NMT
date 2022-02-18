import argparse
from numpy import argmax
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def read(input_file1, input_file2):
    file1 = open(input_file1, 'r', encoding="utf8")
    Lines1 = file1.readlines()
    file2 = open(input_file2, 'r', encoding="utf8")
    Lines2 = file2.readlines()
    target = []
    predicted = []
    for i in range(len(Lines1)):
        target.append(Lines1[i].replace('\n', " "))
    for i in range(len(Lines1)):
        predicted.append(Lines2[i].replace('\n', " "))

    file1.close()
    file2.close()

    return target, predicted

def bleu(target, predicted):
    scores = []
    for i in range(len(target)):
        val = sentence_bleu([target[i].split()], predicted[i].split(), weights=(0.25, 0.25, 0.25, 0.25))
        scores.append(val)

    scores = np.array(scores)
    bleu_score = np.mean(scores)
    print('bleu score for given model {}'.format(bleu_score*100))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        '--input_file1', dest='input_file1', metavar='inputDirectory1', help='dataset directory', required=True)
    requiredNamed.add_argument(
        '--input_file2', dest='input_file2', metavar='inputDirectory2', help='dataset directory', required=True)
    args = parser.parse_args()
    input_file1 = args.input_file1
    input_file2 = args.input_file2

    target, predicted = read(input_file1, input_file2)
    bleu(target, predicted)