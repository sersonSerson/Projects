import pickle
import numpy as np
from colorama import Fore, Style
import os
import re
import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def load_objects():
    encoder_model = load_model('Data/enc_model_collab.h5', compile=False)
    decoder_model = load_model('Data/dec_model_collab.h5', compile=False)
    with open('Data/tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    return encoder_model, decoder_model, tokenizer


def encode_seq(text, tokenizer):
    text = 'StartPlaceholder ' + text + ' EndPlaceholder'
    enc_seq = tokenizer.texts_to_sequences([text])
    enc_seq = pad_sequences(enc_seq, maxlen=max_seq_length, padding='post')
    enc_seq = to_categorical(enc_seq, num_classes=total_words)
    return enc_seq


def decode_seq(seq):
    words_list = \
        tokenizer.sequences_to_texts(np.argmax(seq, axis=-1).reshape(-1, 1))
    result = ' '.join(words_list).strip()
    return result


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def numbers_regex(length, ordinal=False):
    from num2words import num2words
    numbers_list = \
        [num2words(number, ordinal=ordinal) for number in range(length)]
    numbers_regex = \
        '|'.join([f'^{number_text}\s|\s{number_text}\s|\s' \
                     f'{number_text}$' for number_text in numbers_list])
    return numbers_regex


def preprocess_user_text(text):
    # Remove contractions
    text = decontracted(text)

    # Remove numbers and ordinals
    placeholder = ' NumberText '
    text = re.sub(numbers_regex(100, ordinal=False), placeholder, text)
    placeholder = ' OrdinalText '
    text = re.sub(numbers_regex(100, ordinal=True), placeholder, text)

    # Remove numbers and ordinals
    placeholder = ' NumberDigits '
    text = re.sub('^\d+\s|\s\d+\s|\s\d+$', placeholder, text)

    for c in string.punctuation + '\n' + '\r':
       text = text.replace(c, " ")
    return text


def predict(test_sample, encoder_model, decoder_model):
    encoder_state_h, encoder_state_c = encoder_model.predict(test_sample)
    h_prev = \
        np.array([0 for i in range(total_words)]).reshape(1, 1, total_words)
    pred = []
    for i in range(max_seq_length):
        pred_output, encoder_state_h, encoder_state_c = \
            decoder_model.predict([h_prev, encoder_state_h, encoder_state_c])
        decoded_output = decode_seq(pred_output[0][0])
        if decoded_output == 'endplaceholder':
            break

        pred.append(decoded_output)
        h_prev = pred_output

    return pred


encoder_model, decoder_model, tokenizer = load_objects()
max_seq_length = 22
total_words = len(tokenizer.word_index) + 1


print(Fore.MAGENTA + 'Welcome to the chatbot! \n'
                     'Type "quit" to exit \n' + Style.RESET_ALL)

while True:
    print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end='')
    user_text = input()

    if str.lower(user_text) == 'quit':
        print(Fore.GREEN + 'ChatBot:' + Style.RESET_ALL, 'Bye!')
        break
    user_text = preprocess_user_text(user_text)
    encoded_text = encode_seq(user_text, tokenizer)
    prediction_list = predict(encoded_text, encoder_model, decoder_model)
    prediction = ' '.join(prediction_list)
    print(Fore.GREEN + 'ChatBot:' + Style.RESET_ALL, f'{prediction}')
