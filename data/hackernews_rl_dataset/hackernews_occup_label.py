# from openai import OpenAI
# from datasets import load_dataset
# import os
# import time
# import logging

# os.environ['OPENAI_API_KEY'] = 'sk-proj-wytgexAwJcp5t83DAygxT3BlbkFJ7CUiLYJrN8CKQXYF4P7m'

# # Set up logging
# logging.basicConfig(filename='classification.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load your dataset from the Hugging Face Hub
# dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered")["train"]

# # Define a function to classify text with retry mechanism
# def classify_job_description(example, max_retries=3, retry_delay=5):
#     client = OpenAI()
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "user", "content": f"Classify the following job description into one of these categories: accountant, attorney, journalist, professor, software engineer:\n{example['text']}. Respond with only using the category name."}
#                 ]
#             )
#             result = response.choices[0].message.content
#             print(result)
#             logging.info(f"Classification result: {result}")
#             return {'biasinbios_occupation': result.strip()}
#         except Exception as e:
#             error_message = f"Error occurred: {str(e)}. Retrying in {retry_delay} seconds..."
#             print(error_message)
#             logging.error(error_message)
#             retries += 1
#             time.sleep(retry_delay)
    
#     # If all retries are exhausted, raise the exception
#     error_message = "Failed to classify job description after multiple retries."
#     logging.error(error_message)
#     raise Exception(error_message)

# classify_job_description(dataset[0])

# # Apply the classification function to each example in the dataset
# updated_dataset = dataset.map(classify_job_description, batched=False)

# # Upload the updated dataset to Hugging Face Hub (manual process or using huggingface_hub library)
# updated_dataset.push_to_hub("buseskorkmaz/hiring_w_q_context_256_filtered_biasinbios_occ", private=True)

# from transformers import pipeline
# from datasets import load_dataset
# import os
# import time
# import pickle  # for saving the occupations list

# # Load the dataset from the Hugging Face Hub
# dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered")["train"]

# # Initialize LLaMA model
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# def classify_job_description(text):
#     prompt = f"Classify the following job description into one of these categories: accountant, attorney, journalist, professor, software engineer. Respond with only using the category name:\n{text}"

#     # try:
#     encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt')

#     input_ids = encoded_input['input_ids']
#     attention_mask = encoded_input['attention_mask']
    
#     output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=70)
#     return {'biasinbios_occupation': tokenizer.decode(output[0], skip_special_tokens=True)}
#     # except Exception as e:
#     #    raise Exception(f"Failed to classify job description after multiple retries due to {e}")
    

# occupations = []
# for i in range(len(dataset)):
#     item = dataset[i]['text']
#     result = classify_job_description(item)
#     occupations.append(result['biasinbios_occupation'])

# # except Exception as e:
# #     print(f"An error occurred during processing: {e}")
# #     # Optionally save the partial list to a file for recovery
# #     with open('partial_occupations.pkl', 'wb') as f:
# #         pickle.dump(occupations, f)
# #     print("Partial occupations list saved.")

# # Proceed only if there were no interruptions
# if len(occupations) == len(dataset):
#     updated_dataset = dataset.add_column("biasinbios_occ", occupations)
#     updated_dataset.push_to_hub("buseskorkmaz/hiring_w_q_context_256_filtered_biasinbios_occ", private=True)
# else:
#     print("The dataset was not updated due to an error during processing.")

# from transformers import pipeline
# from datasets import load_dataset
# import os
# import time
# import pickle
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import logging
# import sys

# def setup_logging(log_file):
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )

# def classify_job_description(text, tokenizer, model):
#     prompt = f"Classify the following job description into one of these categories: accountant, attorney, journalist, professor, software engineer. Respond with only using the category name:\n{text}"
#     encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt')
#     input_ids = encoded_input['input_ids']
#     attention_mask = encoded_input['attention_mask']
#     output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=70)
#     return {'biasinbios_occupation': tokenizer.decode(output[0], skip_special_tokens=True)}

# def process_batch(batch_index, dataset, tokenizer, model):
#     # batch = dataset.select(range(batch_index * batch_size, (batch_index + 1) * batch_size))
#     batch = dataset.select(range(5700, 5709))
#     logging.info(f"Processing batch {batch_index + 1}/{num_batches}")
#     occupations = []
#     for item in batch['text']:
#         result = classify_job_description(item, tokenizer, model)
#         occupations.append(result['biasinbios_occupation'])
#     batch = batch.add_column("biasinbios_occupations",occupations)
#     logging.info(f"Finished processing batch {batch_index + 1}/{num_batches}")
#     return batch

# def main(batch_index):
#     setup_logging(f"processing_batch_{batch_index}.log")
#     logging.info("Loading dataset...")
#     dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered")["train"]
#     logging.info("Loading LLaMA model...")
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#     logging.info("Processing batch...")
#     occupations = process_batch(batch_index, dataset, tokenizer, model)
#     logging.info("Saving processed batch...")
#     occupations.save_to_disk(f"occupations_processed_batch_{batch_index}")
#     logging.info("Processing completed.")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <batch_index>")
#         sys.exit(1)
#     batch_index = int(sys.argv[1])
#     num_batches = 50
#     batch_size = len(load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered")["train"]) // num_batches
#     main(batch_index)


from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def classify_job_description(text, pipeline):
    messages = [
        {"role": "system", "content": "You are an AI assistant that classifies job descriptions into categories."},
        {"role": "user", "content": f"Classify the following job description into one of these categories: accountant, attorney, journalist, professor, software engineer. Respond with only using the category name:\n{text}"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=70,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return {'biasinbios_occupation': outputs[0]["generated_text"][len(prompt):].strip()}

def process_batch(batch_index, dataset, pipeline):
    # batch = dataset.select(range(batch_index * batch_size, (batch_index + 1) * batch_size))
    batch = dataset.select(range(5700 + (batch_index * batch_size), 5700 + (batch_index + 1) * batch_size))
    logging.info(f"Processing batch {batch_index + 1+520}/{num_batches+520}")
    occupations = []
    for item in batch['text']:
        result = classify_job_description(item, pipeline)
        occupations.append(result['biasinbios_occupation'])
    batch = batch.add_column("biasinbios_occupations", occupations)
    logging.info(f"Finished processing batch {batch_index + 1+520}/{num_batches+520}")
    return batch

def main(batch_index):
    setup_logging(f"processing_batch_{batch_index+520}.log")
    logging.info("Loading dataset...")
    dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered")["train"]
    logging.info("Loading LLaMA model...")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        # model_kwargs={"torch_dtype": torch.bfloat16},
        # device="cuda",
    )
    logging.info("Processing batch...")
    occupations = process_batch(batch_index, dataset, pipe)
    logging.info("Saving processed batch...")
    occupations.save_to_disk(f"occupations_processed_batch_{batch_index+520}")
    logging.info("Processing completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <batch_index>")
        sys.exit(1)
    batch_index = int(sys.argv[1])
    num_batches = 1
    batch_size = (len(load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered")["train"]) - 5700) // num_batches
    main(batch_index)