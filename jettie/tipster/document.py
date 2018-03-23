from .annotation import Annotation

class Document(object):
  def __init__(self, stg=None, doc=None):
    if doc:
      self.init_with_doc(doc)
    elif stg:
      self.text = stg
      self.annotationsByStart = dict()
      self.annotationsByEnd = dict()
      self.annotationsByType = dict()
    else:
      self.text = ""
      self.annotationsByStart = dict()
      self.annotationsByEnd = dict()
      self.annotationsByType = dict()

  def init_with_doc(self, doc):
    self.text = doc.text
    # TODO
    pass

  def clear(self):
    self.text = ""
    self.annotationsByStart = dict()
    self.annotationsByEnd = dict()
    self.annotationsByType = dict()

  def get_text_by_span(self, span):
    return self.text[span.start:span.end]

  def get_text_by_ann(self, ann):
    return self.get_text_by_span(ann.span)

  def normalized_text_by_span(self, span):
    return self.text[span.start:span.end].strip()

  def normalized_text_by_ann(self, ann):
    return self.normalized_text_by_span(ann.span)

  def get_lentgh(self):
    return len(self.text)

  def get_char_at(self, p):
    return self.text[p]

  def clear_annotations(self):
    self.annotationsByStart = dict()
    self.annotationsByEnd = dict()
    self.annotationsByType = dict()

  def add_annotation(self, ann):
    start = ann.span.start
    if start not in self.annotationsByStart.keys():
      self.annotationsByStart[start] = list()
    self.annotationsByStart[start].append(ann)
    end = ann.span.end
    if end not in self.annotationsByEnd.keys():
      self.annotationsByEnd[end] = list()
    self.annotationsByEnd[end].append(ann)
    type = ann.type
    if type not in self.annotationsByType.keys()
      self.annotationsByType[type] = list()
    self.annotationsByType[type].append(ann)

    # TODO set document to span
    return ann

  def annotate(self, start, end, att):
    # TODO add start end att tro annotation
    ann = Annotation()
    return self.add_annotation(ann)

  def remove_annotation(self, ann):
    start = ann.span.start
    if start in self.annotationsByStart.keys():
      self.annotationsByStart[start].remove(ann)

    end = ann.span.end
    if end in self.annotationsByEnd.keys():
      self.annotationsByEnd[end].remove(ann)

    type = ann.type
    if type in self.annotationsByType.keys():
      self.annotationsByType.remove(ann)
    return

  def remove_annotations_by_type(self, t):
    if self.annotationsByType.keys():
      for ann in self.annotationsByType[t]:
        self.remove_annotation(ann)
    return

  # return list

  def get_annotations_by_start(self, start):
    if start not in self.annotationsByStart.keys():
      return None
    return self.annotationsByStart[start]

  def get_annotations_by_start_type(self, start, t):
    if start not in self.annotationsByStart.keys():
      return None
    ann_list = list()
    for ann in self.annotationsByStart[start]:
      if ann.type == t:
        ann_list.append(ann)
    return ann_list

  def get_annotations_by_start_types(self, start, ts):
    if start not in self.annotationsByStart.keys():
      return None
    ann_list = list()
    for ann in self.annotationsByStart[start]:
      for t in ts:
        if ann.type == t:
          ann_list.append(ann)
    return ann_list

  def get_annotations_by_end(self, end):
    if end not in self.annotationsByEnd.keys():
      return None
    return self.annotationsByEnd[end]

  def get_annotations_by_end_type(self, end, t):
    if end not in self.annotationsByEnd.keys():
      return None
    ann_list = list()
    for ann in self.annotationsByEnd[end]:
      if ann.type == t:
        ann_list.append(ann)
    return ann_list

  def get_annotations_by_type(self, t):
    if t not in self.annotationsByType.keys():
      return None
    return self.annotationsByType[t]

  def get_annotations_by_type_span(self, t, span):
    if t not in self.annotationsByType.keys():
      return None
    ann_list = list()
    for ann in self.annotationsByType[t]:
      # TODO wihin function for span
      if ann.span.within(span):
        ann_list.append(ann)
    return ann_list