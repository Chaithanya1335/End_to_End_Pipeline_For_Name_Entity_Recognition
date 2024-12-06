from src.Exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from src.Utils import load_files
from dataclasses import dataclass
import pickle
import os
import sys

@dataclass
class ModelConfig:
    model_path = os.path.join('artifacts','model.keras')
class ModelTraining:
    def __init__(self) -> None:
        self.ModelConfig = ModelConfig()
    def initiate_model_training(self,x_train,x_test,y_train,y_test):

        logging.info("Model Training started")
        try:
            word_to_index_path = os.path.join("artifacts","word_to_index.json")
            word_to_index=load_files(word_to_index_path)

            index_to_tag_path = os.path.join("artifacts","index_to_tag.json")
            index_to_tag=load_files(index_to_tag_path)

            max_len_path = os.path.join("artifacts","maxlen.txt")

            with open(max_len_path,'r') as file:
                max_len = file.read()
            
            max_len = int(max_len)

            # Build LSTM model
            vocab_size = len(word_to_index) + 1 # +1 for padding
            embedding_dim = 50
            lstm_units = 100
            num_classes = len(index_to_tag) + 1


            model = Sequential()
            model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
            model.add(LSTM(lstm_units, return_sequences=True))  # Return sequences for sequence tagging
            model.add(Dense(num_classes, activation='softmax')) # Use softmax for multi-class classification

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test)) # Adjust epochs and batch size

            logging.info("Model Training Finished")

            y_pred = model.predict(x_test)

            loss,accuracy = model.evaluate(y_test,y_pred) 

            logging.info(f"Accuracy Achieved:{accuracy} Loss:{loss}")

            
            model.save(self.ModelConfig.model_path)

            logging.info("Model Saved")
            return (loss,accuracy,self.ModelConfig.model_path)
        except Exception as e:
            raise CustomException(e,sys)