from SIDD import logger
from SIDD.components.data_preparation import DataPreparation
from SIDD.components.loss_and_metrics import LossAndMetrics
from SIDD.components.model_building import ModelBuilding
from SIDD.components.model_training import ModelTraining
from SIDD.config.configuration import ConfigurationManager


STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        # Data
        data_config = config.get_data_preparation_config()
        data = DataPreparation(data_config)
        train_loader, val_loader = data.get_dataloaders()

        # Model
        model_config = config.get_model_building_config()
        model = ModelBuilding(model_config).build_model()

        # Loss
        loss_config = config.get_loss_metrics_config()
        criterion = LossAndMetrics(loss_config).get_bce_dice_loss()

        # Trainer
        train_config = config.get_model_training_config()

        trainer = ModelTraining(
            config=train_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion
        )

        import torch

        print("=" * 50)
        print("PyTorch Version :", torch.__version__)
        print("CUDA Available  :", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU             :", torch.cuda.get_device_name(0))
            print("Current Device  :", torch.cuda.current_device())
        else:
            print("Running on CPU")
        print("=" * 50)

        trainer.train()
    
if __name__ == '__main__': 
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        raise e
