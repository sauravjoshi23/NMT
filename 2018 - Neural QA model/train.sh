#!/usr/bin/env bash
cd nmt
python -m nmt.nmt --src=en --tgt=sparql --vocab_prefix=../data/place_30/vocab --dev_prefix=../data/place_30/dev --test_prefix=../data/place_30/test --train_prefix=../data/place_30/train --out_dir=../data/place_30/model --num_train_steps=2 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=bleu 
cd ..