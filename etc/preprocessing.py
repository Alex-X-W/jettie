import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base = 0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m,s)

def timeSince(since,percent):
	now = time.time()
	s = now -since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

INPUTSET = list()
OUTPUTSET = list()

MAX_LENGTH = 0

START = 0
EOS = 1
class Lang:
	def __init__(self,name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "START",1:'EOS'}
		self.n_words = 2

	def index_words(self,sentence):
		for word in sentence.split(' '):
			self.index_word(word)

	def index_word(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1
		return 

class LSTMEncoder(nn.Module):
	def __init__(self,embedding_dim,hidden_dim,vocab_size,embedding):
		super(LSTMEncoder, self).__init__()
		self.hidden_dim = hidden_dim
		self.embedding = embedding
		self.lstm1 = nn.LSTM(embedding_dim,hidden_dim)
		self.dropout1 = nn.Dropout(p=0.2)
		self.lstm2 = nn.LSTM(hidden_dim,hidden_dim)
		self.dropout2 = nn.Dropout(p=0.2)
		self.lstm3 = nn.LSTM(hidden_dim,hidden_dim)
	def init_hidden(self):
		return (Variable(torch.zeros(1,1,self.hidden_dim)),
			Variable(torch.zeros(1,1,self.hidden_dim)))

	def forward(self,input,hidden):
		embedding = self.embedding(input).view(1,1,-1)
		output = embedding
		output, hidden = self.lstm1(output,hidden)
		output = self.dropout1(output)
		output, hidden = self.lstm2(output,hidden)
		output = self.dropout2(output)
		output, hidden = self.lstm3(output,hidden)
		return output, hidden

class LSTMADecoder(nn.Module):
	def __init__(self, embedding_dim,hidden_dim,embedding,output_size,max_length = MAX_LENGTH): # how to decide output_size?
		super(LSTMADecoder,self).__init__()
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.output_size = output_size
		self.embedding = nn.Embedding(self.output_size,self.embedding_dim)
		self.max_length = max_length
		# print(self.max_length)

		self.lstm1 = nn.LSTM(hidden_dim,hidden_dim)
		self.dropout = nn.Dropout(p=0.2)
		self.lstm2 = nn.LSTM(hidden_dim,hidden_dim)
		self.lstm3 = nn.LSTM(hidden_dim,hidden_dim)
		
		self.out = nn.Linear(hidden_dim,output_size) # output vocab? required?
		self.softmax = nn.LogSoftmax(dim =1)

		self.attn = nn.Linear(self.hidden_dim + self.embedding_dim, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim)

	def forward(self,input,hidden,encoder_outputs):
		# print(input)
		embedding = self.embedding(input).view(1,1,-1)
		# print(hidden[0].size(),embedding[0].size())
		output = embedding
		# print(embedding.size,hidden.size)
		attn_weights = self.softmax(self.attn(torch.cat((embedding[0],hidden[0][0]),1)))
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
		
		output = torch.cat((embedding[0],attn_applied[0]),1)
		output = self.attn_combine(output).unsqueeze(0)
		output = F.relu(output)
		# print(output.size())
		output,hidden = self.lstm1(output,hidden)
		output = self.dropout(output)
		output,hidden = self.lstm2(output,hidden)
		output = self.dropout(output)
		output,hidden = self.lstm3(output,hidden)
		output = self.softmax(self.out(output[0])) # change here

		# decoder_output,decoder_hidden,decoder_attention
		return output, hidden, attn_weights

	def init_hidden(self):
		return Variable(torch.zeros(1,1,self.hidden_dim))

def create_word_embeding(input_lang,path,embedding_dim):
	word2index = input_lang.word2index
	# print(word2index)
	embedding_file = open(path+'/glove.6B.'+str(embedding_dim)+'d.txt','r')
	embedding_lines = embedding_file.readlines()
	embedding_dict = dict()
	for line in embedding_lines:
		line = line.strip()
		lines = line.split(' ')
		if not len(lines) == (embedding_dim+1):
			pass
		else:
			embedding_dict[lines[0]] = []
			features_list = lines[1:]
			for f in features_list:
				embedding_dict[lines[0]].append(float(f))
	embedding_word2dict = {} 
	for key in word2index.keys():
		if key.lower() in embedding_dict.keys():
			embedding_word2dict[word2index[key]] = embedding_dict[key.lower()]
		else:
			print(key)
	embedding_word2list = []
	index = len(word2index)
	for i in range(index+2):
		if i not in embedding_word2dict.keys():
			listtmp = []
			for j in range(embedding_dim):
				listtmp.append(0)
			embedding_word2list.append(listtmp)
		else:
			embedding_word2list.append(embedding_word2dict[i])
	numpy_word2list = np.asarray(embedding_word2list)

	# frozen weight!!
	embed = nn.Embedding(index+2, embedding_dim)
	embed.weight.data.copy_(torch.from_numpy(numpy_word2list))
	# print(numpy_word2list)
	embed.weight.requires_grad = False
	return embed
	# print (numpy_word2list.shape)


def indexes_from_sentence(lang,sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang,sentence):
	indexes = indexes_from_sentence(lang,sentence)
	indexes.append(1)
	# var = Variable(torch.LongTensor(indexes).view(-1,1,1))
	var = Variable(torch.LongTensor(indexes).view(-1,1))
	return var

def variables_from_pair(pair,input_lang,output_lang):
	input_variable = variable_from_sentence(input_lang,pair[0])
	# print(pair[1])
	target_variable = variable_from_sentence(output_lang,pair[1])
	return input_variable, target_variable

def read_langs(path):
	path = "ontonotes.train.conll"
	train = open(path,'r')
	trainData = train.readlines()
	tag_linear = ''
	inputlist = list()
	outputlist = list()
	input_lines = list()
	output_lines = list()
	pairs = list()
	inputline = None
	outputline = None

	MAX_LENGTH = 0
	for i in range(300):
		td = trainData[i]
	# for td in trainData:
		tdstrip = ' '.join(td.split()) # clean
		tdlist = tdstrip.split(' ') # split
		if not len(tdlist) > 6:
			tags = []
			head = []
			tag_tmp = None
			start = False
			for c in tag_linear:
				if c == '(':
					if tag_tmp and not tag_tmp == '':
						tags.append(tag_tmp)
					if start:
						head.append(tag_tmp.replace('(',''))
					tag_tmp = c
					start = True
				elif c == ' ':
					if tag_tmp and not tag_tmp == '':
						tags.append(tag_tmp)
					if start:
						head.append(tag_tmp.replace('(',''))
					start = False
					tag_tmp = ''
				elif c == ')':
					tag_tmp = c + head[-1]
					tags.append(tag_tmp)
					head = head[:-1]
					tag_tmp = ''
				else:
					tag_tmp += c
			if inputline:
				outputline = ' '.join(tags)
				output_lines.append(outputline)
				input_lines.append(inputline)
				pairtmp = (inputline,outputline)
				if len(tags) > MAX_LENGTH:
					MAX_LENGTH = len(tags)
				pairs.append(pairtmp)
				inputline = None
				outputline = None
				
			tag_linear = ''
		else:
			token = tdlist[3]
			postag = tdlist[4]
			parsertag = tdlist[5]
			if postag.upper() == postag.lower():
				postagstr = ' '+postag + ' '
			else:
				postagstr = ' XX '
			parsertmp = parsertag.replace('*',postagstr)
			tag_linear += parsertmp
			if inputline:
				inputline += ' '
				inputline += token
			else:
				inputline = token

	input_lang = Lang('sentence')
	output_lang = Lang('parsertag')

	for pair in pairs:
		# print(pair[0])
		input_lang.index_words(pair[0])
		output_lang.index_words(pair[1])

	return input_lang,output_lang,pairs, MAX_LENGTH
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
	input_variable = variable_from_sentence(input_lang,sentence)
	input_length = input_variable.size()[0]
	encoder_hidden = encoder.init_hidden()

	encoder_outputs = Variable(torch.zeros(max_length,encoder.hidden_dim))
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_variable[ei],encoder_hidden)
		# print(encoder_output)
		encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
	decoder_input = Variable(torch.LongTensor([[START]]))
	decoder_hidden = encoder_hidden
	decoded_words = []
	decoder_attentions = torch.zeros(max_length,max_length)
	for di in range(max_length):
		decoder_output,decoder_hidden,decoder_attention = decoder(
			decoder_input, decoder_hidden, encoder_outputs)
		decoder_attentions[di] = decoder_attention.data
		topv, topi = decoder_output.data.topk(1)
		# print(topi.size())
		ni = topi[0][0]
		if ni == EOS:
			decoded_words.append('EOS')
			break
		else:
			decoded_words.append(output_lang.index2word[ni])

		decoder_input = Variable(torch.LongTensor([[ni]]))
	return decoded_words, decoder_attentions[:di+1]

def evaluateRandomly(encoder,decoder, max_length, n=10):
	for i in range(n):
		pair = random.choice(pairs)
		print('>',pair[0])
		print('=',pair[1])
		output_words, attentions = evaluate(encoder,decoder,pair[0],max_length)
		output_sentence = ' '.join(output_words)
		print('<',output_sentence)
		print('')

teacher_forcing_ratio = 0.5

def train(input_variable,target_variable,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):
	encoder_hidden = encoder.init_hidden()
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	encoder_outputs = Variable(torch.zeros(max_length,encoder.hidden_dim))
	loss = 0
	# print(input_variable,input_length,target_length)
	for ei in range(input_length):
		# print (input_variable[ei])
		encoder_output, encoder_hidden = encoder(input_variable[ei],encoder_hidden)
		# print(encoder_output[0][0].size(),encoder_outputs[ei].size())
		encoder_outputs[ei] = encoder_output[0]

	decoder_input = Variable(torch.LongTensor([[START]]))
	decoder_hidden = encoder_hidden
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		for di in range(target_length):
			# input,hidden,encoder_outputs
			decoder_output,decoder_hidden,decoder_attention = decoder(
				decoder_input,decoder_hidden,encoder_outputs)
			loss += criterion(decoder_output, target_variable[di])
			decoder_input = target_variable[di]
	else:
		for di in range(target_length):
			decoder_output, decoder_hidden,decoder_attention = decoder(
				decoder_input,decoder_hidden,encoder_outputs)
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0]

			decoder_input = Variable(torch.LongTensor([[ni]]))
			loss += criterion(decoder_output,target_variable[di])
			if ni == EOS:
				break 
	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0]/target_length

def trainIters(input_lang,output_lang,pairs,max_length, encoder,decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01):
	start = time.time()
	plot_losses = []
	print_loss_total = 0
	plot_loss_total = 0

	encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),lr = learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(),lr = learning_rate)
	training_pairs = [variables_from_pair(random.choice(pairs),input_lang,output_lang) for i in range(n_iters)]

	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = training_pairs[iter-1]
		input_variable = training_pair[0]
		target_variable = training_pair[1]

		loss = train(input_variable,target_variable,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total/print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start,iter/n_iters),iter,iter/n_iters*100,print_loss_avg))
			evaluateRandomly(encoder, decoder, max_length)

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0
	showPlot(plot_losses)

if __name__ == "__main__":
	path = "ontonotes.train.conll"
	input_lang, output_lang, pairs, MAX_LENGTH = read_langs(path)
	# print(MAX_LENGTH)
	path = "/Users/jiayunyu/Desktop/Spring2018/NLP/Assignment 7/glove.6B"
	embedding = create_word_embeding(input_lang,path,50)

	hidden_dim = 256 # embedding_dim,hidden_dim,vocab_size,embedding
	print(input_lang.n_words)
	encoder = LSTMEncoder(50, hidden_dim, input_lang.n_words, embedding)
	# embedding_dim,hidden_dim,embedding
	decoder = LSTMADecoder(50, hidden_dim,embedding,output_lang.n_words,MAX_LENGTH)

	trainIters(input_lang, output_lang, pairs,MAX_LENGTH,encoder,decoder, 5000, print_every = 500)
	evaluateRandomly(encoder, decoder, MAX_LENGTH)
	# input_variable, _ =variables_from_pair(pairs[0],input_lang,output_lang)
	# input_length = input_variable.size()
	# print(embedding(input_variable).view(input_length[1],1,50))