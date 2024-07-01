import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
import hydra
from omegaconf import DictConfig, OmegaConf
# import bias.load_objects
# from eval.iql_inference import generations
from eval.bc_inference import generations

@hydra.main(config_path="../../../config/hackernews", config_name="eval_policy_bc")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    generations(cfg)

if __name__ == "__main__":
    main()
