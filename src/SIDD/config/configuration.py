from SIDD.utils.common import read_yaml, create_directories
from SIDD.constants import *
from SIDD.entity.config_entity import DataIngestionConfig, EDAConfig

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
            num_classes=self.config.image.num_classes,
            order=self.config.rle.order
        )

        return eda_config
