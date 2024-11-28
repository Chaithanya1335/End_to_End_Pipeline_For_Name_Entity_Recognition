from src.Exception import CustomException
from src.logger import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import spacy
import os
import sys
import pandas as pd

class DataTransformation:
    def __init__(self) -> None:
        pass
    def initiate_dataTransformation(self,data_path):
        try:
            nlp = spacy.load('en_core_web_sm')

            logging.info("Data Transformation started")

            logging.info("loading Text data")

            data = pd.read_csv(data_path)

            logging.info("Data Loaded")

            def process_text(text):
                docs = nlp(text)
                tokens = []
                tags = []
                for token in docs:
                    tokens.append(token.text)
                    tags.append(token.pos_)
                return tokens,tags

            word_to_index = {}
            tag_to_index = {}
            index_to_word = {}
            index_to_tag = {}

            all_tokens = [token for sentence_tokens in data['Text'].apply(lambda text: process_text(text)[0]).to_list() for token in sentence_tokens]
            all_tags = [tag for sentence_tags in data['Text'].apply(lambda text: process_text(text)[1]).to_list() for tag in sentence_tags]

            for i, token in enumerate(set(all_tokens)):
                word_to_index[token] = i + 1
                index_to_word[i+1] = token

            for i, tag in enumerate(set(all_tags)):
                tag_to_index[tag] = i + 1
                index_to_tag[i+1] = tag

            logging.info("Converting Senteces into Numerical sequences")
            # Convert sentences to numerical sequences
            X = [[word_to_index.get(token, 0) for token in sentence_tokens] for sentence_tokens in data['Text'].apply(lambda Sentence: process_text(Sentence)[0]).to_list()]
            y = [[tag_to_index[tag] for tag in sentence_tags] for sentence_tags in data['Text'].apply(lambda Sentence: process_text(Sentence)[1]).to_list()]

            logging.info("Padding the sequences")
            # Pad sequences to have the same length
            max_len = max(len(seq) for seq in X)
            X = pad_sequences(X, maxlen=max_len, padding='post')
            y = pad_sequences(y, maxlen=max_len, padding='post')

            # One-hot encode the labels
            y = tf.keras.utils.to_categorical(y, num_classes=len(tag_to_index) + 1)

            logging.info("Splitting The Data")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Data Transformation Completed")

            return (X_train,X_test,y_train,y_test,max_len,word_to_index,tag_to_index)
        except Exception as e:
            raise CustomException(e,sys)
        