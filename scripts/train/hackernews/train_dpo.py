import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
import hydra
from omegaconf import DictConfig, OmegaConf
# import bias.load_objects
from train.dpo_train_loop import train

@hydra.main(config_path="../../../config/hackernews", config_name="train_dpo")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
