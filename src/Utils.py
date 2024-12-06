import os
import sys
import pickle
import json
import spacy
from src.Exception import CustomException
from src.logger import logging


    
def process_text(text):
     try:
        nlp = spacy.load('en_core_web_sm')
        docs = nlp(text)
        tokens = []
        tags = []
        for token in docs:
            tokens.append(token.text)
            tags.append(token.pos_)
        return tokens,tags
     except Exception as e:
            raise CustomException(e,sys)
     

def load_model(path):
    try:
        logging.info("Loading The Model")
        return pickle.load(open(path,'rb'))
    except Exception as e:
        raise CustomException(e,sys)
    
def load_files(path):
    try:
        logging.info("Loading Files")
        with open(path,'r') as file:
            return json.load(file)
    
    except Exception as e:
        raise CustomException(e,sys) 
    
def save_files(path,file):
    try:
        logging.info("Saving Files")
        with open(path,'w') as fil:
            return json.dump(file,fil)
    
    except Exception as e:
        raise CustomException(e,sys) 
            

