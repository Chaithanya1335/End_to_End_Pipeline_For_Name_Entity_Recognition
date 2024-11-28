from src.Exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from dataclasses import dataclass
import pickle
import os
import sys

@dataclass
class ModelConfig:
    model_path = os.path.join('artifacts','model.h5')
class ModelTraining:
    def __init__(self) -> None:
        self.ModelConfig = ModelConfig()
    def initiate_model_training(self,x_train,x_test,y_train,y_test,max_len,word_to_index,tag_to_index):

        logging.info("Model Training started")
        try:
            # Build LSTM model
            vocab_size = len(word_to_index) + 1 # +1 for padding
            embedding_dim = 50
            lstm_units = 100
            num_classes = len(tag_to_index) + 1


            model = Sequential()
            model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
            model.add(LSTM(lstm_units, return_sequences=True))  # Return sequences for sequence tagging
            model.add(Dense(num_classes, activation='softmax')) # Use softmax for multi-class classification

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test)) # Adjust epochs and batch size

            logging.info("Model Training Finished")

        

            loss,accuracy = model.evaluate(x_test,y_test) 

            logging.info(f"Accuracy Achieved:{accuracy} Loss:{loss}")

            
            with open(self.ModelConfig.model_path,'wb') as file:
                pickle.dump(model,file)
            logging.info("Model Saved")
            return (loss,accuracy)
        except Exception as e:
            raise CustomException(e,sys)