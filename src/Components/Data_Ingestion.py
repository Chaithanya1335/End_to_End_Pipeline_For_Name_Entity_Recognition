from src.Exception import CustomException
from src.logger import logging
from src.Components.DataTransformation import DataTransformation
from src.Components.Model import ModelTraining
from src.Components.Model import ModelConfig
from src.Components.Predict import Predict
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
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            
            data = pd.read_csv(r'D:\projects\End_to_End_Pipeline_For_Name_Entity_Recognition\ner.csv')

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
    text = "Hi iam Gnana Chaithanya, aspiring Data Scientist Taking coaching at 360digitmg"
    text_path = DataIngestion().initiate_data_Ingestion()
    x_train,x_test,y_train,y_test= DataTransformation().initiate_dataTransformation(text_path)
    loss,accuracy,model_path = ModelTraining().initiate_model_training(x_train,x_test,y_train,y_test)
    Name_Entitys = Predict().predict_data(text=text)
