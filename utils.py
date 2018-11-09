import unicodedata
import os
import re
import random
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
import sys

# Normalize text by mapping non-ascii characters to approximate ascii. e.g., beyonc'{e} becomes beyonce
def normalize_unicode(text):
  #return text.encode('ascii', 'ignore')
  return unicodedata.normalize('NFD', text).encode('ascii', 'ignore')

# Standard word tokenizer.
_treebank_word_tokenize = TreebankWordTokenizer().tokenize

def word_tokenize(text, language='english'):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    if sys.version_info[0] < 3:
        return [token for token in _treebank_word_tokenize(text)]
    else:
        return [token for token in _treebank_word_tokenize(text.decode("UTF-8"))]

def get_ngrams(n, tokens, separator=" "):
  if n == 0:
    return [" ".join(tokens)]

  # extract each n-token sequence from entire sequence of tokens
  ngrams = []
  for i, token in enumerate(tokens):
    # first k-gram at position k-1
    if i >= n - 1:
      ngrams.append(separator.join(tokens[i - n + 1:i + 1]))
  return ngrams

def get_vector(embedding, term):
  if term in embedding:
    return embedding[term]
  elif term.title() in embedding:
    return embedding[term.title()]
  elif term.lower() in embedding:
    return embedding[term.lower()]
  elif term.upper() in embedding:
    return embedding[term.upper()]
  return None


def get_word_vector(entity_model, word):
    if type(entity_model) == tuple:
        vocab, emb = entity_model
        wid = vocab[word]
        return emb[wid]
    else:
        return get_vector(entity_model, word)

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def invert_dict(dict):
  """
  Convert a dict (string->int) into a list of strings (ie, dict value becomes list index)
  :param dict:
  :return:
  """
  dict_inv = [""] * (max(dict.values()) + 1)
  if sys.version_info[0] < 3:
      for word, index in dict.iteritems():
        dict_inv[index] = word
  else:
      for word, index in dict.items():
          dict_inv[index] = word
  return dict_inv

def clean_text(question):
  """
  Prepare question text for tokenization: lowercase, remove punctuation, and remove episode numbers (these are added during Spark pipeline)
  e.g., "Who plays in Seinfeld: The Contest S10E8?" ==> "who plays in seinfeld the contest"
  :param question: string representing question (not tokenized)
  :return: string representing cleaned up question, ready for tokenization
  """
  question = re.sub("[\.\t\,\:;\(\)\?\!]", " ", question.lower(), 0, 0)
  return re.sub("s\d+e\d+", "", question, 0, 0)

def unsplit_query(query, qrepr, vocab_inv):
  """
  Regenerate query from core elements, depending on the query representation.
  :param query: query as string
  :param qrepr: query representation (e.g., word or char).
  :param is_already_tokenized: use True if ``query'' was generated using vocab_inv (possibly using defeaturize),
          so we do not need to preprocess text
  :return:
  """
  PAD_WORD_INDEX = 0
  if qrepr == "word":
    return " ".join([vocab_inv[int(w)] for w in query if w != PAD_WORD_INDEX])
  elif qrepr == "char":
    return "".join([vocab_inv[int(w)] for w in query if w != PAD_WORD_INDEX])
  elif qrepr.endswith("gram"):
    query_str = ""
    for w in query:
      if w != PAD_WORD_INDEX:
        if len(query_str) == 0:
            query_str = vocab_inv[int(w)]
        else:
            query_str += vocab_inv[int(w)][-1]
    return query_str[1:-1] # remove # mark in the beginning and end position.
  else:
      raise Exception("Unrecognized representation %s!" % qrepr)


def split_sent(sent, qrepr, ngram_size=3):
    """
    Split sentence into core elements, depending on the query representation.
    :param sent: sent as string
    :param qrepr: query representation (e.g., word or char).
    :param is_already_tokenized: use True if ``query'' was generated using vocab_inv (possibly using defeaturize),
            so we do not need to preprocess text
    :return:
    """
    if qrepr == "word":
        return word_tokenize(sent)
    elif qrepr == "char":
        cs = list(sent)
        return [c for i, c in enumerate(cs) if i == 0 or c != " " or cs[i - 1] != " "]
    elif qrepr.endswith("gram"):
        if sys.version_info[0] < 3:
            return get_ngrams(ngram_size, split_sent("#"+sent+"#", "char"), separator="")
        else:
            return get_ngrams(ngram_size, split_sent("#" + sent.decode("utf-8") + "#", "char"), separator="")
    else:
        raise Exception("Unrecognized representation %s!" % qrepr)