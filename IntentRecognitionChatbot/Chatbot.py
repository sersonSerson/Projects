import pickle
import numpy as np
import json
import random
from colorama import Fore, Style

import nltk
from nltk.corpus import names

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_objects():
    model = load_model('Models/chatbot_model.h5')
    with open('Models/label_encoder.pickle', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    with open('Models/tokenizer.pickle', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    return model, label_encoder, tokenizer


def predict_intent(text, tokenizer, label_encoder):
    test_seq = tokenizer.texts_to_sequences(text)
    test_pad_seq = pad_sequences(test_seq, maxlen=max_seq_length,
                                 padding='post')
    intent = label_encoder.inverse_transform([
        np.argmax(model.predict(test_pad_seq))])
    return intent[0]


def responses():
    with open('Data/Intent.json') as file:
        data = json.load(file)

    responses_dict = {}
    for x in data['intents']:
        responses_dict[x['intent']] = x['responses']

    return responses_dict


def find_name(user_text):
    tkn = nltk.word_tokenize(user_text)
    names_list = [token for token in tkn if token in names.words()]
    name = None
    if len(names_list):
        name = names_list[0]
    return name


def give_response(intent, responses_dict, name=None):
    responses = responses_dict[intent]
    n_responses = len(responses)
    response_n = random.randint(0, n_responses - 1)
    response = responses[response_n]
    if name:
        response = response.replace('<HUMAN>', name)
    return response

max_seq_length = 20
model, label_encoder, tokenizer = load_objects()
responses_dict = responses()
name = None

print(Fore.MAGENTA + 'Welcome to the chatbot! \n'
      'Type "quit" to exit \n' + Style.RESET_ALL)

while True:
    print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end='')
    user_text = input()

    if str.lower(user_text) == 'quit':
        break

    new_name = find_name(user_text)
    if new_name:
        name = new_name
    intent = predict_intent([user_text], tokenizer, label_encoder)
    response = give_response(intent, responses_dict, name)
    print(Fore.GREEN + 'ChatBot:' + Style.RESET_ALL, f'{response} ({intent})')
