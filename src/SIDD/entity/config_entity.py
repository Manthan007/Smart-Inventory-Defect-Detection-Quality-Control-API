from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class EDAConfig:
    train_csv: Path
    train_images_path: Path
    test_images_path: Path
    height: int
    width: int
    num_classes: int

@dataclass(frozen=True)
class DataPreparationConfig:
    train_csv_path: Path
    train_images_path: Path
    batch_size: int
    height: int
    width: int
    num_classes: int
    num_workers: int