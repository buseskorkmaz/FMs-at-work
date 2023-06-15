from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import os
import json

def load_dataset(dataset_name = "stereoset copy.csv"):

    print("os cwd",os.getcwd())

    dataset = pd.read_csv(dataset_name)
    dataset_size = dataset.shape[0]
    dataset["gold_label"] = dataset["gold_label"].multiply(-1)
    dataset.drop(columns=["id", "target", "bias_type"], inplace=True)
    dataset.dropna(inplace=True)
    print("Dataset shape:", dataset.shape)
    dataset = dataset[~dataset["context"].str.fullmatch("The ", case=True)]
    print("Dataset shape:", dataset.shape)
    dataset.reset_index(drop=True, inplace=True)
    # dataset = tokenize(dataset=dataset)
    dataset.rename(columns={"context":"state", "sentence":"action", "gold_label":"q_value"}, inplace=True)
    dataset.to_csv("stereoset.csv", index=False)
    # train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test  = train_test_split(dataset[dataset.columns[:-1]], dataset[dataset.columns[-1]], test_size=0.1, random_state=42, shuffle=False)

    X_train, X_eval, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False) # 0.2 x 0.9 = 0.18

    train_idxs = X_train.index.to_list()
    test_idxs =  X_test.index.to_list()
    eval_idxs = X_eval.index.to_list()

    print(len(train_idxs))
    print(len(test_idxs))
    print(len(eval_idxs))


    # train_idxs_j = json.dumps(train_idxs)
    # test_idxs_j = json.dumps(test_idxs)
    # val_idxs_j = json.dumps(val_idxs)

    with open('train_idxs.json', 'w') as f:
        json.dump(train_idxs, f)
    
    with open('test_idxs_j.json', 'w') as f:
        json.dump(test_idxs, f)
    
    with open('eval_idxs_j.json', 'w') as f:
        json.dump(eval_idxs, f)


    # train_idxs.to_json("train_idxs.json")
    # test_idxs.to_json("test_idxs.json")
    # val_idsx.to_json("val_idsx.json")

    return 


def tokenize(dataset, model_name="EleutherAI/gpt-neo-1.3B"):

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    context_ids_list = []
    sentence_ids_list = []
    for i in range(len(dataset)):
        context_ids_list.append(tokenizer.encode(dataset.iloc[i]["context"], return_tensors='pt'))
        sentence_ids_list.append(tokenizer.encode(dataset.iloc[i]["sentence"], return_tensors='pt'))

    dataset["context"] = context_ids_list
    dataset["sentence"] = sentence_ids_list

    return dataset



def load_model():

    return


load_dataset()