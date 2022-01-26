#!/usr/bin/env python
"""
Usage: python build_vocab.py data.en > vocab.en
"""
import numpy as np
import sys

x_text = list()
with open(sys.argv[1]) as f:
    for line in f:
        x_text.append(str(line[:-1]))

vocabulary = set()

lang = sys.argv[1].split('.')[-1].lower()
# print lang

if lang == "sparql":

    for x in x_text:
        for t in x.split(" "):
            vocabulary.add(t)

else:  # any other language

    for x in x_text:
        for t in x.split(" "):
            vocabulary.add(t)

    # split also by apostrophe

    to_remove = set()
    to_add = set()
    for t0 in vocabulary:
        if "'" in t0:
            to_remove.add(t0)
            for t1 in t0.split("'"):
                to_add.add(t1)
    for t0 in to_remove:
        vocabulary.remove(t0)
    for t0 in to_add:
        vocabulary.add(t0)

with open("data/place_30/vocab."+lang, "w") as out_file:
   for v in vocabulary:
        out_file.write(v)
        out_file.write("\n")
