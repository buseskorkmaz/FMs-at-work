from datasets import load_dataset
import re
from transformers import pipeline
import multiprocessing
import torch
import argparse

print(multiprocessing.cpu_count())

def preprocess_text(row):
    text = str(row["text"])
    text = re.sub("\n", '', text)
    text = re.sub('"', '', text)

    first_step = re.sub(r'http\S+|www.\S+', '', text)
    second_step = re.sub(r'<a href=+|>+', '', first_step)

    return {"clean_text": second_step}

def not_none(example):
    return example['text'] is not None

def parse_job_description(
    row, 
    question:str,
    ) -> str:

    # Initialize the question-answering pipeline
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name, max_length=512, device=torch.device('cuda:0'))

    # The context from which the model will extract information
    context = row["text"]

    # Questions you want the model to answer
    questions = {
        "loc": "What are the locations mentioned in the text?",
        "tech": "What technologies and libraries are mentioned in the text?",
        "title": "What is the job title mentioned in the text?",
        "comp_name": "What is the company name mentioned in the text?",
        # "What technical skills are mentioned in the text?",
    }

    # Ask each question
    result = nlp(question=questions[question], context=context)
    # print(result["answer"])
    # print(f"Question: {question}")
    # print(f"Answer: {result['answer']}")

    # Print the question and the predicted answer
    # print("Question:", question)
    # print("Answer:", answer)
    if question ==  "loc":
        key = "location"
    elif  question == "title":
        key = "title" 
    elif question =="comp_name":
        key= "company"
    else:
        key = "technologies"

    return {key: result["answer"]}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")
    parser.add_argument("--job", type=str, required=True, help="The job name to process (comp_name/title)")

    args = parser.parse_args()
    index = args.index
    job_name = args.job

    print(f"Job {job_name} is starting....")

    dataset = load_dataset("dansbecker/hackernews_hiring_posts", split='hiring')
    print(dataset)
    # hiring_df  = dataset.to_pandas()

    num_batches = 24
    batched_datasets = []
    length_of_dataset = len(dataset)

    for i in range(num_batches):
        start_index = int((i * length_of_dataset) / num_batches)
        end_index = int(((i + 1) * length_of_dataset) / num_batches)
        batched_datasets.append(dataset.shard(num_batches, i))

    dataset = batched_datasets[index] 

    # dropna
    dataset = dataset.filter(not_none)

    # remove links
    # dataset = dataset.map(preprocess_text, num_proc=8)

    dataset = dataset.map(lambda x: parse_job_description(x, job_name))

    if job_name =="comp_name":
        print("Example company_name", dataset["company"][10])
    else:
        print("Example title", dataset["title"][10])

    dataset.save_to_disk(f"processed_hiring_{job_name}_{index}")
