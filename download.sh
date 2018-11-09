#!/bin/bash

cd data; mkdir word2vec
cd word2vec

echo ">>>Download word2vec from Google Drive..."
gdrive download 0B7XkCwpI5KDYNlNUTTlSS21pQmM

cd ../twitter-v0
echo ">>>Download IDF weights of word ngrams..."
gdrive download 0B1EhxQ7GBJdsZTVmcFVMcDY1RWM
tar -xf collection_word_idf.json.tar
rm collection_word_idf.json.tar

echo ">>>Download IDF weights of char ngrams..."
gdrive download 0B1EhxQ7GBJdsbXdROGZQYzV5cFU
tar -xf collection_ngram_idf.json.tar
rm collection_ngram_idf.json.tar

echo ">>>Build trec_eval tool..."
cd ../..
tar -xf trec_eval.8.1.tar.gz
cd trec_eval.8.1
# suppress make warnings
make --ignore-errors 2> make.log
rm make.log

echo "Done."
