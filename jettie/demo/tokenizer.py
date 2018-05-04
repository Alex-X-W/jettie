# jettie architecture
from jettie.tipster import *
from jettie.tokenizer.base_tokenizer import BaseTokenizer

# tokenizer implementation
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import PunktSentenceTokenizer


class SimpleTokenizer(BaseTokenizer):
  def __init__(self, doc, tok_anno_name='token', sent_anno_name='sentence'):
    self.doc = doc
    self.tok_anno_name = tok_anno_name
    super(SimpleTokenizer, self).__init__(self.doc, self.tok_anno_name)
    self.sent_anno_name = sent_anno_name
    self.sentTokenizer = PunktSentenceTokenizer()
    self.treebankTokenizer = TreebankWordTokenizer()

  def tokenize(self, start_pos=0, text=None):
    if text is None: # if text not given, assume it spans to the end of doc
      text = self.doc.get_text_by_span(Span(start_pos, self.doc.get_length()))

    # add sentence annotations
    sents = self.sentTokenizer.tokenize(text)
    sent_spans = self.sentTokenizer.span_tokenize(text)
    for sent, sent_span in zip(sents, sent_spans):
      self.doc.annotate(start_pos + sent_span[0], start_pos + sent_span[1], self.sent_anno_name)

    # use sentence annotation to retrieve sentences
    for sent in self.doc.get_annotations_by_type(self.sent_anno_name):
      toks = self.treebankTokenizer.tokenize(self.doc.get_text_by_ann(sent))
      spans = self.treebankTokenizer.span_tokenize(self.doc.get_text_by_ann(sent))
      sent_base = sent.span.start
      for tok, tok_span in zip(toks, spans):
        span = Span(tok_span[0] + sent_base, tok_span[1] + sent_base)
        self.doc.annotate(span.start, span.end, self.tok_anno_name)


if __name__ == '__main__':
  sample_txt = 'This is a sample text. It has two sentences.'
  true_toks = 'This is a sample text . It has two sentences .'.split()

  doc = Document(sample_txt)

  tok_name = 'tok'
  sent_name = 'sent'
  simpleTokenizer = SimpleTokenizer(doc, tok_anno_name=tok_name, sent_anno_name=sent_name)
  simpleTokenizer.tokenize(start_pos=0)

  print('%10s %10s %10s' % ('response', 'key', 'hit'))
  for tok, true_tok in zip(doc.get_annotations_by_type(tok_name), true_toks):
    print('%10s %10s %10s' % (doc.get_text_by_ann(tok), true_tok, (doc.get_text_by_ann(tok) == true_tok).__repr__()))
