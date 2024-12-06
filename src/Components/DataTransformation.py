import spacy
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.Exception import CustomException
from src.logger import logging
from src.Utils import save_files
import sys
import os
from collections import Counter
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    word_to_index_path = os.path.join("artifacts","word_to_index.json")
    index_to_tag_path = os.path.join("artifacts","index_to_tag.json")
    max_len_path = os.path.join("artifacts",'maxlen.txt')
    



class DataTransformation:
    def __init__(self) -> None:
        # Load spaCy model once during initialization
        self.nlp = spacy.load('en_core_web_sm')
        self.DataTransformationConfig = DataTransformationConfig() 
    
    def initiate_dataTransformation(self, data_path):
        try:
            logging.info("Data Transformation started")
            logging.info("Loading text data")
            
            # Load the dataset
            data = pd.read_csv(data_path)
            logging.info("Data Loaded")
            
            # Initialize mappings
            word_to_index = {}
            tag_to_index = {}
            index_to_word = {}
            index_to_tag = {}
            
            logging.info("Extracting Tokens and POS Tags")
            
            # Use a Counter to track token and tag frequencies
            all_tokens = []
            all_tags = []
            
            for text in data['Text']:
                tokens, tags = self.process_text(text)
                all_tokens.extend(tokens)
                all_tags.extend(tags)

            # Create word and tag dictionaries using Counter for efficiency
            word_counts = Counter(all_tokens)
            tag_counts = Counter(all_tags)
            
            word_to_index = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
            tag_to_index = {tag: i+1 for i, (tag, _) in enumerate(tag_counts.items())}
            index_to_word = {i+1: word for i, (word, _) in enumerate(word_counts.items())}
            index_to_tag = {i+1: tag for i, (tag, _) in enumerate(tag_counts.items())}
            
            logging.info("Tokens and POS Tags Extracted")
            
            # Convert sentences to numerical sequences
            X = [[word_to_index.get(token, 0) for token in sentence_tokens] for sentence_tokens in data['Text'].apply(lambda sentence: self.process_text(sentence)[0]).to_list()]
            y = [[tag_to_index.get(tag, 0) for tag in sentence_tags] for sentence_tags in data['Text'].apply(lambda sentence: self.process_text(sentence)[1]).to_list()]
            
            logging.info("Padding the sequences")
            
            # Pad sequences to have the same length
            max_len = max(len(seq) for seq in X)
            X = pad_sequences(X, maxlen=max_len, padding='post')
            y = pad_sequences(y, maxlen=max_len, padding='post')
            
            with open(self.DataTransformationConfig.max_len_path,'w') as file:
                file.write(str(max_len))

            # One-hot encode the labels
            
            y = tf.keras.utils.to_categorical(y, num_classes=len(tag_to_index) + 1)

            logging.info("Saving word_to_index and index_to_tag")

            save_files(self.DataTransformationConfig.word_to_index_path,word_to_index)

            save_files(self.DataTransformationConfig.index_to_tag_path,index_to_tag)         
            
            logging.info("Splitting the Data")
            

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            
            
            logging.info("Data Transformation Completed")
            
            return (X_train, X_test, y_train, y_test)
        
        except Exception as e:
            # Provide detailed exception info
            raise CustomException(e, sys)
    
    def process_text(self, text):
        try:
            # Tokenize the text and extract POS tags
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            tags = [token.pos_ for token in doc]
            return tokens, tags
        except Exception as e:
            raise CustomException(e, sys)
