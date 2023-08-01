from datasets import load_dataset

hiring_dataset = load_dataset("buseskorkmaz/hackernews_new_q_values_10", split='train')
print(hiring_dataset)