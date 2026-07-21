from SIDD.components.data_preparation import DataPreparation
from SIDD.components.loss_and_metrics import LossAndMetrics
from SIDD.components.model_building import ModelBuilding
from SIDD.components.model_evaluation import ModelEvaluation
from SIDD.config.configuration import ConfigurationManager



class ModelEvaluationPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        data_config = config.get_data_preparation_config()

        data = DataPreparation(data_config)

        _, val_loader = data.get_dataloaders()

        

        model_config = config.get_model_building_config()

        model = ModelBuilding(model_config).build_model()



        loss_config = config.get_loss_metrics_config()

        loss_metrics = LossAndMetrics(loss_config)

        criterion = loss_metrics.get_bce_dice_loss()

        dice_metric = loss_metrics.get_dice_score()

        iou_metric = loss_metrics.get_iou_score()


        evaluation_config = config.get_model_evaluation_config()

        evaluator = ModelEvaluation(

            config=evaluation_config,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            dice_metric=dice_metric,
            iou_metric=iou_metric

        )

        evaluator.evaluate()
        evaluator.visualize_predictions(num_images=5)
        evaluator.evaluate_thresholds()


from SIDD import logger

STAGE_NAME = "Model Evaluation"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = ModelEvaluationPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e