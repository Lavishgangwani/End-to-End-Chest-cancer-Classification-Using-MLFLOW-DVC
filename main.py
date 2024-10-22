from ChestCancerClassifier import logger
from ChestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<") 

except Exception as e:
      raise e