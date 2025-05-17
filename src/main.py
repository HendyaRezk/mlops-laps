import sys
import os
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess_data import preprocess_data
from src.model.train_model import train_model
from src.evaluate import evaluate

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  
    preprocess_data(cfg)  
    train_model(cfg)  
    evaluate(cfg)  

if __name__ == "__main__":
    main()