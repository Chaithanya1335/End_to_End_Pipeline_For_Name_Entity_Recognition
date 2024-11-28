from src.Exception import CustomException
from src.logger import logging
from src.Components.DataTransformation import DataTransformation
from src.Components.Model import ModelTraining
from src.Components.Model import ModelConfig
from dataclasses import dataclass
import pandas as pd
import sys 
import os

@dataclass
class DataIngestionConfig:
    text_path = os.path.join('artifacts','text.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.DataIngestionConfig = DataIngestionConfig()
    def initiate_data_Ingestion(self):

        logging.info("Data Ingestion started")

        try:
            data = pd.read_csv(r'D:\projects\Name_entity_Recognition\ner.csv')

            logging.info("Data Readed as DataFrame")

            logging.info("Extracting only Text from Data")

            os.makedirs(os.path.dirname(self.DataIngestionConfig.text_path),exist_ok=True)

            text ={ "Text" : data['Sentence']}
            
            text = pd.DataFrame(text)

            text.to_csv(self.DataIngestionConfig.text_path,index=False,header=True)

            logging.info("Text Extracted from data")

            logging.info("DataIngestion Completed")

            return self.DataIngestionConfig.text_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    text_path = DataIngestion().initiate_data_Ingestion()
    x_train,x_test,y_train,y_test,max_len,word_to_index,tag_to_index = DataTransformation().initiate_dataTransformation(text_path)
    loss,accuracy = ModelTraining().initiate_model_training(x_train,x_test,y_train,y_test,max_len,word_to_index,tag_to_index)