from datasets import load_dataset, DatasetDict
import json

dataset =  load_dataset("buseskorkmaz/hackernews_hiring_w_q")["train"]

# Add an index to the dataset
dataset = dataset.add_column('index', list(range(len(dataset))))

# Split the dataset into train and test
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
test_dataset = dataset['test']

# Further split the train dataset into train and validation
train_dataset = train_dataset.train_test_split(test_size=0.111)  # 5% of 90% is 4.5%, so we need to split off approximately 5.556% of the train_dataset to get 5% of the original dataset
train_dataset_final = train_dataset['train']
val_dataset = train_dataset['test']

print(train_dataset_final)
print(test_dataset)
print(val_dataset)

# Save the indices of each split
with open('train_idxs.json', 'w') as f:
    json.dump(train_dataset_final['index'], f)

with open('eval_idxs.json', 'w') as f:
    json.dump(val_dataset['index'], f)

with open('test_idxs.json', 'w') as f:
    json.dump(test_dataset['index'], f)
