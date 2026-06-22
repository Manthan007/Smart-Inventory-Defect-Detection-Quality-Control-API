from src.SIDD.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.SIDD import logger

STAGE_NAME = "data ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    raise e