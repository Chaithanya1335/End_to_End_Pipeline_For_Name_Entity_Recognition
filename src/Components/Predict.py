from src.Exception import CustomException
from src.logger import logging
from src.Utils import process_text,load_files
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from src.Utils import load_model
import os
import sys
import numpy as np

class Predict:
    def __init__(self) -> None:
        pass
    def predict_data(self,text):
        try:
            logging.info("Prediction started")
            model_path = os.path.join("artifacts","model.keras")
            model = tf.keras.models.load_model(model_path)
            tokens,_= process_text(text)

            logging.info("Loading The Word_to_index and tag_index")

            word_to_index_path = os.path.join("artifacts","word_to_index.json")
            word_to_index = load_files(word_to_index_path)

            index_to_tag_path = os.path.join("artifacts","index_to_tag.json")
            index_to_tag = load_files(index_to_tag_path)


            max_len_path = os.path.join("artifacts","maxlen.txt")

            with open(max_len_path,'r') as file:
                max_len = file.read()
            
            max_len = int(max_len)

            logging.info("Converting tokens into Numerical Sequences")
            numerical_sequences = [word_to_index.get(token,0) for token in tokens]

            logging.info("Padding The Numerical Sequences")
            paded_tokens = pad_sequences([numerical_sequences], maxlen=max_len, padding='post')
            Predictions = model.predict([paded_tokens])[0]

            logging.info("Getting Tags Indexes")
            prediction_tags_index = np.argmax(Predictions,axis=1)
            predicted_tags = [index_to_tag.get(index,'O') for index in prediction_tags_index]

            logging.info("Getting Name Entity's From given text")
            names = []
            for i in range(len(tokens)):
                if predicted_tags[i]=='PROPN':
                    names.append(tokens[i])
            logging.info(f"Name Entity's Extracted : {names}")
            return names
        except Exception as e:
            raise CustomException(e,sys)
        