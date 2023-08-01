import argparse
import pickle as pkl
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src'))
import dill


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    args = parser.parse_args()

    with open(args.eval_file, 'rb') as f:
        d = dill.load(f)

    rs = [sum(map(lambda x: x[2], item[1])) for item in d['eval_dump']['results']]
    mean_r = np.mean(rs)
    std_r = np.std(rs)
    st_err_r = std_r / np.sqrt(len(rs))
    ent = [-item for item in d['eval_dump']['entropies']]
    mean_ent = np.mean(ent)
    std_ent = np.std(ent)
    st_err_ent = std_ent / np.sqrt(len(ent))
    print(d['config'])
    print(f'reward: {mean_r} +- {st_err_r}')
    print(f'entropy: {mean_ent} +- {st_err_ent}')
    print(len(ent), len(rs))

