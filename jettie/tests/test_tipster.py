"""
running a naive test using Tipster:
1) read from a sample text and store into an instance of `Document` class
2) tokenize and add annotation `token`
"""

import unittest

from jettie.tipster import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import PunktSentenceTokenizer


class TestTipster(unittest.TestCase):
  def test_document(self):
    """
    test basic use of Tipster's Document class, detailed tests TODO
    """
    # we are using naive text sample here, and a primitive PunktSentenceTokenizer to split the sentences
    # more advanced and practical sentence splitter is needed to handle abbreviations
    sample_txt = 'This is a sample text. It has two sentences.'
    true_toks = 'This is a sample text . It has two sentences .'.split()

    # add sentence annotations
    doc = Document(sample_txt)
    senttokenizer = PunktSentenceTokenizer()
    sents = senttokenizer.tokenize(sample_txt)
    sent_spans = senttokenizer.span_tokenize(sample_txt)
    for sent, sent_span in zip(sents, sent_spans):
      doc.annotate(sent_span[0], sent_span[1], 'sentence')

    # add token annotations
    treebanktokenizer = TreebankWordTokenizer()
    # use sentence annotation to retrieve sentences
    for sent in doc.get_annotations_by_type('sentence'):
      toks = treebanktokenizer.tokenize(doc.get_text_by_ann(sent))
      spans = treebanktokenizer.span_tokenize(doc.get_text_by_ann(sent))
      sent_base = sent.span.start
      for tok, tok_span in zip(toks, spans):
        span = Span(tok_span[0] + sent_base, tok_span[1] + sent_base)
        doc.annotate(span.start, span.end, 'token')

    # check if all tokens are correct
    for tok, true_tok in zip(doc.get_annotations_by_type('token'), true_toks):
      self.assertTrue(doc.get_text_by_ann(tok) == true_tok)


if __name__ == '__main__':
  unittest.main()
