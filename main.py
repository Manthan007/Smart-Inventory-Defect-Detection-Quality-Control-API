from SIDD.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from SIDD import logger
from SIDD.pipeline.stage_02_model_training import ModelTrainingPipeline
from SIDD.pipeline.stage_03_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "data ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    raise e


STAGE_NAME = "Model Training stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    raise e


STAGE_NAME = "Model Evaluation"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = ModelEvaluationPipeline()
    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e