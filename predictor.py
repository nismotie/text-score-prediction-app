import library as lib
from library import f1_m, precision_m, recall_m
import string
import pickle
import pandas as pd
import numpy as np
import csv

from keras import layers;
from keras import optimizers;
from keras import models;
from keras import losses;
from keras import metrics;
from keras.utils.np_utils import to_categorical;
from keras.preprocessing.text import Tokenizer;
from keras.preprocessing.sequence import pad_sequences;
from keras.utils.np_utils import to_categorical;
from keras import backend as K;

dependencies = {
    'f1_m': f1_m,
    'precision_m': precision_m,
    'recall_m': recall_m
}

model = models.load_model('conv_embedding_model_lrg.h5', custom_objects=dependencies)
tk = pickle.load(open("tk_20k_vocab_200_words.pkl", "rb" ))
review_len = model.layers[0].input_shape[1]
def predictor():
    text = input("Please enter your review here: ") 
    if len(text)<4:
        print('Please try again.')
        predictor()
    input_sw = lib.remove_stopwords(text, join=True)
    list_input = [input_sw]
    tokenized_input = tk.texts_to_sequences(list_input)
    padded_input = pad_sequences(tokenized_input, maxlen=review_len)
    prediction = int(model.predict_classes(padded_input)) + 1
    user_rating = 6
    while user_rating > 5:
        user_rating = int(input("Give a star rating out of five: "))
    print(f'The predicted star rating for your review is: {prediction}')
    with open('collected_app_data.csv', mode='a') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([prediction, user_rating, text])
        data.close()
    predictor() 
predictor()

