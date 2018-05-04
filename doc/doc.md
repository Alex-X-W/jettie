# Jettie

*a light weight python information extraction framework, a descendant of NYU's JET IE system*



### Motivation

Tipster architecture good for

- pipelining
- systematic experiment on IE tasks
- alignment of annotations which might facilitates future visualization work.

In this project, we consider a minimal architecture of Tipster.



### Code Structure

![](/Users/xuanwang/Documents/18Spring/nlp/jettie/doc/code_structure.jpg)



### Basic Design

#### Primary architecture: Tipster

Tipster is implemented in the subpackage of `jettie`, which is the core of ensuring annotations are well aligned. While python is dynamically typed language, much effort can be saved from writing getter and setter methods.

#### NLP component interface

In order for NLP task components to comply with Tipster, a natural way to do is require the customized components to provide methods which annotate and align text in a Tipster way. While in Java, such requirements can be implemented through interface, but in python, we do it in a base class and ask the customized component to inherent from the base class, and implement the methods defined in the base class. To see an example, see the `BaseTokenizer` class defined in `./tokenizer/base_tokenizer.py`, and its inherent subclass `SimpleTokenizer` defined in `./demo/tokenizer.py`.



### Tests

We performed unittest with automatic unit test tool `nose`, for an unittest case example please see `/tests/test_tipster.py`. Since we would still develop and use the project code in our own future work, writting automatic tests can make our life a little easier and neater.

One can run all the unittests at the `./jettie/` directorie by issueing `$ nosetests .`, and ideal output indicating test cases passed should look like `./doc/unittest.png`.

