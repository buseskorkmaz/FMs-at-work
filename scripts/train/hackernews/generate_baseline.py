import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def generate_conversational_response(row, max_length=256):
    """
    Generate a conversational response using GPT-2 based on the given prompt.
    
    Parameters:
    - prompt (str): The input prompt for the model.
    - model: The pretrained GPT-2 model.
    - tokenizer: The GPT-2 tokenizer.
    - max_length (int): Maximum length of the generated text.
    - temperature (float): Sampling temperature for generation.
    - top_k (int): Top-k sampling for generation.
    - top_p (float): Top-p (nucleus) sampling for generation.
    
    Returns:
    - str: Generated response.
    """
   
    # Sample prompt from your dataset
    prompt = row["prompt"]

    # Split the string
    parts = prompt.split('Based on the original description,')

    # original_description = parts[0].strip()
    new_description = 'Based on the original description,' + parts[1].strip()

    # print("Original Description:")
    # print(original_description)
    # print("\nNew Description:")
    # print(new_description)

    # Frame the prompt in a dialogue format
    dialogue_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:\n{new_description}\n###Response:\n"""
    
    # Ensure input tensors are on the GPU
    input_ids = tokenizer.encode(dialogue_prompt, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the Bot's response from the generated text
    response = generated_text.split("Response:\n")[-1].strip()
    print(response)
    
    return {"baseline_text": response}

if __name__ == "__main__":

    hiring_dataset = load_dataset("buseskorkmaz/hackernews_new_q_values_10", split="train")
    print(hiring_dataset)
    # print(hiring_dataset[10])
    # print("=="*25)
    # prompt = hiring_dataset[10]["prompt"]

    # # Split the string
    # parts = prompt.split('Based on the original description,')

    # original_description = parts[0].strip()
    # new_description = 'Based on the original description,' + parts[1].strip()

    # print("Original Description:")
    # print(original_description)
    # print("\nNew Description:")
    # print(new_description)
    tokenizer = AutoTokenizer.from_pretrained("vicgalle/gpt2-open-instruct-v1")
    model = AutoModelForCausalLM.from_pretrained("vicgalle/gpt2-open-instruct-v1")

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)  # Move the model to the GPU

    hiring_dataset = hiring_dataset.map(generate_conversational_response)
    print(hiring_dataset[512])
    hiring_dataset.push_to_hub("buseskorkmaz/baseline_text", private=True)