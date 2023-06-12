import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append("/Users/busesibelkorkmaz/Desktop/Bias-ILQL/")
import hydra
from omegaconf import DictConfig, OmegaConf
# import bias.load_objects
from train.bc_train_loop import train

@hydra.main(config_path="../../../config/bias", config_name="train_reward_model")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
