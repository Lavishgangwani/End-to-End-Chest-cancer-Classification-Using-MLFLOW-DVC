from ChestCancerClassifier import logger
from ChestCancerClassifier.components.data_ingestion import DataIngestion
from ChestCancerClassifier.entity.config_entity import DataIngestionConfig
from ChestCancerClassifier.config.configuration import ConfigurationManager


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zipfile()
        except Exception as e:
            raise e
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<") 
    except Exception as e:
        raise e