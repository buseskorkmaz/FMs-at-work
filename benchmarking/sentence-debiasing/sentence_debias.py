import torch
import transformers
import json
from models.sentence_debias_model import SentenceDebiasGPT2LMHeadModel
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GemmaTokenizer, GemmaForCausalLM

def generate_text(prompt, model, mode,tokenizer=None, max_length=256):
    # Set the pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if mode == "mistral":
        prompt = "<s> [INST] " + prompt + " [/INST]"
        encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    
    elif mode == "llama":
        encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    
    else:
        encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', max_length=max_length, padding_side='left', truncation=True)

    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    
    output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def read_prompts(file_path):
    prompts = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['prompt']['text'])
    return prompts

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description="Debias job descriiptions with SentenceDebias."
    )

    parser.add_argument(
        "--mode",
        action="store",
        type=str,
        default="gender",
        help="Bias direction. Options are gender and race",
    )

    parser.add_argument(
        "--index",
        action="store",
        type=int,
    )

    args = parser.parse_args()
    mode = args.mode
    index = args.index

    print(f'{mode} mode selected ..')

    if mode == 'llama':
        model_path = 'meta-llama/Llama-2-7b-chat-hf'
        # model_path = 'openlm-research/open_llama_7b'

        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, device_map='auto',
    )
    
    elif mode == 'gemma':
        model_path = 'google/gemma-2b'

        tokenizer = GemmaTokenizer.from_pretrained(model_path)
        model = GemmaForCausalLM.from_pretrained(
            model_path, device_map='auto',
        )

    elif mode == 'mistral':
        model_path = 'mistralai/Mistral-7B-Instruct-v0.2'

        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, device_map='auto',
        )

    elif mode == 'sentence_debias':
        # Assuming bias_direction is your pre-computed bias direction tensor
        # Load it here, for example:
        bias_direction = torch.load(f'$HOME/sentence-debiasing/subspaces/subspace_m-GPT2Model_c-gpt2_t-{mode}.pt')
        model = SentenceDebiasGPT2LMHeadModel('gpt2', bias_direction)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    output_file = f'$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_{mode}_{index}.jsonl'
    prompts = read_prompts('$HOME/FMs-at-work/benchmarking/sentence-debiasing/debias_prompts-v2.jsonl')
    num_batches = 20
    batch_size = len(prompts) // num_batches

    prompts = prompts[(index)*batch_size: (index+1)*batch_size]
    for idx, prompt in enumerate(prompts):
        generated_text = generate_text(prompt, model, mode, tokenizer)
        data = {
            'prompt': prompt,
            'generated_text': generated_text
        }
        write_to_jsonl(output_file, data)
        print(f"{idx}/{len(prompts)} done ...")

    print("Debiasing completed!")
