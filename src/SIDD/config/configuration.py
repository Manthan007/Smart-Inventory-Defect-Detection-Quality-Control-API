from SIDD.utils.common import read_yaml, create_directories
from SIDD.constants import *
from SIDD.entity.config_entity import DataIngestionConfig, DataPreparationConfig, EDAConfig, LossMetricsConfig, ModelBuildingConfig, ModelEvaluationConfig, ModelTrainingConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    

    def get_eda_config(self) -> EDAConfig:
        config = self.config.data

        eda_config = EDAConfig(
            train_csv=config.train_csv,
            train_images_path=config.train_images_path,
            test_images_path=config.test_images_path,
            height=self.config.image.height,
            width=self.config.image.width,
            num_classes=self.config.image.num_classes
        )

        return eda_config
    

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        params = self.params.data_preparation

        data_preparation_config = DataPreparationConfig(
            train_csv_path=config.train_csv_path,
            train_images_path=config.train_images_path,
            batch_size=params.batch_size,
            height=config.height,
            width=config.width,
            num_classes=config.num_classes,
            num_workers=params.num_workers
        )

        return data_preparation_config
    
    def get_model_building_config(self) -> ModelBuildingConfig:
        config = self.config.model_building
        params = self.params.model_building

        model_building_config = ModelBuildingConfig(
            root_dir=config.root_dir,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            features=params.features
        )

        return model_building_config
    

    def get_loss_metrics_config(self) -> LossMetricsConfig:
        config = self.config.loss_and_metrics

        loss_metrics_config = LossMetricsConfig(
            smooth=config.smooth
        )

        return loss_metrics_config
    

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.model_training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            batch_size=params.batch_size,
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            lr_scheduler=params.lr_scheduler,
            model_path=Path(config.model_path)
        )

        return model_training_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
            config = self.config.model_evaluation

            create_directories([config.root_dir])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=Path(config.root_dir),
                model_path=Path(config.model_path)
            )

            return model_evaluation_config