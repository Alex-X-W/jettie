class BaseTokenizer(object):
  """ Base class for tokenizer, all tokenizer should inherent from it.

  It ensures the actual tokenizer implementation comply with Tipster, so that the tokens generated from the
  tokenizer implementation can be well aligned.
  """
  def __init__(self, doc, tok_anno_name='token'):
    """
    :param doc: an instance of `tipster.Document`
    :param tok_anno_name: the annotation name assigned to token
    """
    self.tok_anno_name = tok_anno_name
    self.doc = doc

  def tokenize(self, start_pos=0, text=None):
    """ the main function of a tokenizer

    The `start_pos` specifies the starting character position
    of the text the tokenzier takes in, in order to keep things aligned.
    """
    pass