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
    parser.add_argument('--eval_file', type=str, default="/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/eval/frozen_512_hn_no_offload_10_epoch-beta64_awac025_1gpu/eval_logs_16499.pkl")
    args = parser.parse_args()

    with open(args.eval_file, 'rb') as f:
        d = dill.load(f)

    print(d)
    print("=="*50)
    print(d['eval_dump'])
    print(len(d['eval_dump']))
    print([str(item[1]) for item in d['eval_dump']['results']])

    rs = [sum(map(lambda x: x[2], item[1])) for item in d['eval_dump']['results']]
    print(len(rs))
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

# import argparse
# import pickle as pkl
# import numpy as np
# import os
# import sys
# import glob
# import dill

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src'))

# if __name__ == "__main__":
    
#     combined_eval_dump = []
#     combined_config = None
#     for process_id in range(0, 40):
#         for local_process_id in range(0,4):
#             try:
#                 file_path = f"$HOME/FMs-at-work/outputs/hackernews/openllamav2/eval/iql_eval_10epoch-beta05-{process_id}-{local_process_id}/eval_logs.pkl"
#                 with open(file_path, 'rb') as f:
#                     d = dill.load(f)

#                     # Combine eval_dump
#                     combined_eval_dump.extend(d['eval_dump']['results'])

#                     # Assuming all configs are the same, take the config from the first file
#                     if combined_config is None:
#                         combined_config = d['config']
#             except:
#                 print("Missed", process_id, "-", local_process_id)

#     # Now `combined_eval_dump` contains all the results
#     print("=="*50)
#     print(combined_eval_dump)
#     print(len(combined_eval_dump))
#     print([str(item[1]) for item in combined_eval_dump])

#     rs = [sum(map(lambda x: x[2], item[1])) for item in combined_eval_dump]
#     mean_r = np.mean(rs)
#     std_r = np.std(rs)
#     st_err_r = std_r / np.sqrt(len(rs))
    
#     # entropies = [-item for sublist in combined_eval_dump for item in sublist[1]]
#     # mean_ent = np.mean(entropies)
#     # std_ent = np.std(entropies)
#     # st_err_ent = std_ent / np.sqrt(len(entropies))

#     print(combined_config)
#     print(f'reward: {mean_r} +- {st_err_r}')
#     print(len(rs))
#     # print(f'entropy: {mean_ent} +- {st_err_ent}')
#     # print(len(entropies), len(rs))
