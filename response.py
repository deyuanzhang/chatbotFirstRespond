#NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer

#Tensorflow
import numpy as np 
import tflearn
import tensorflow as tf 
import random
import pickle

import json

stemmer = LancasterStemmer()
data = pickle.load(open ("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)

net = tflearn.input_data(shape =[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
#load the saved model
model.load('./model.tflearn')

#create bag of words from user input
def clean_up_sentence(sentence):
    #tokenize pattern
    sentence_words = nltk.word_tokenize(sentence)
    #stem the words
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#bag of words array: 0 or 1 for word in bag
def bow(sentence, words, show_details=False):
    #tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    #bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" %w)
    return(np.array(bag))

context = {}

#response processor
ERROR_THRESHOLD = 0.25
def classify(sentence):
    #generate probabilities from model
    results = model.predict([bow(sentence, words)])[0]
    #filter out prediction below a theshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    #sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    #intent/probability tuple
    return return_list

def response(sentence, userID='123', show_details = False):
    results = classify(sentence)
    #if classification found, find matching intent
    if results:
        #loop long as matches are available to process
        while results:
            for i in intents['intents']:
                #find tag matching first result
                if i['tag'] == results[0][0]:
                    #set context for intent
                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[userID] = i['context_set']
                    #check if intent is contextual and applies to conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print('tag:', i['tag'])
                        return print(random.choice(i['responses']))

            results.pop(0)

