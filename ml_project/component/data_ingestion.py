from ml_project.exception import ProjectException
from ml_project.logger import logging
from ml_project.entity.config_entity import DataIngestionConfig
from ml_project.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from six.moves import urllib
import sys,os
import zipfile 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ProjectException(e,sys)
    

    def download_ml_project_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            os.makedirs(tgz_download_dir,exist_ok=True)

            ml_project_file_name = 'ml_project.zip'

            tgz_file_path = os.path.join(tgz_download_dir, ml_project_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")
            urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            return tgz_file_path

        except Exception as e:
            raise ProjectException(e,sys) from e

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")
            with zipfile.ZipFile(tgz_file_path) as myzip:
                myzip.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

        except Exception as e:
            raise ProjectException(e,sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            ml_project_file_path = os.path.join(raw_data_dir,file_name)


            logging.info(f"Reading csv file: [{ml_project_file_path}]")
            df = pd.read_csv(ml_project_file_path)

            #renaming the columns by replacing '-' hiphen with '_' underscore
            df.rename(columns={'education-num': 'education_num',
                    'marital-status': 'marital_status',
                    'capital-gain':'capital_gain',
                    'capital-loss':'capital_loss',
                    'hours-per-week':'hours_per_week'},
                    inplace=True, errors='raise')

            #renaming the column values by replacing '-' hiphen with '_' underscore
            cat = ['workclass','education','marital_status','occupation','relationship','race','sex','country']
            for i in cat:
                df[i] = df[i].str.replace('-','_')
                df[i] = df[i].str.strip()
            
            #replacing '?' with nan
            df = df.replace(' ?', np.nan)

            # removing duplicates
            df.drop_duplicates(inplace=True)

            # filling null values
            for col in ['workclass', 'occupation', 'country']:
                df[col].fillna(df[col].mode()[0], inplace=True)
            
            df['salary'] = df['salary'].str.replace('<=50K', '0')
            df['salary'] = df['salary'].str.replace('>50K', '1')
            df['salary'] = df['salary'].astype('int')

            #create two different dataframe of majority and minority class 
            df_majority = df[(df['salary']==0)] 
            df_minority = df[(df['salary']==1)] 

            # upsample minority class
            df_minority_upsampled = resample(df_minority,
                                            replace=True,    # sample with replacement
                                            n_samples= df['salary'].value_counts().max(), # to match majority class
                                            random_state=42,
                                            )  # reproducible results

            # Combine majority class with upsampled minority class
            df_upsampled = pd.concat([df_minority_upsampled, df_majority])

            df_upsampled.reset_index(inplace=True,drop=True)
            df_upsampled["binned"] = df_upsampled['salary']

            logging.info(f"Splitting data into train and test")

            print(df_upsampled.shape)
            train_set, test_set, _, _ = train_test_split(df_upsampled.iloc[:,:-1], df_upsampled.iloc[:,-1], test_size = 0.2, random_state = 0, stratify=df_upsampled['binned'])

            print(train_set.shape,test_set.shape)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                train_set.to_csv(train_file_path,index=False)

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_set.to_csv(test_file_path,index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise ProjectException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path =  self.download_ml_project_data()
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise ProjectException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")