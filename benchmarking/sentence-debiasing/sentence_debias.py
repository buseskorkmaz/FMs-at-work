import torch
import transformers
import json
from models.sentence_debias_model import SentenceDebiasGPT2LMHeadModel
import argparse

def generate_text(prompt, model, tokenizer=None, max_length=512):
    # Set the pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', max_length=max_length, padding_side='left', truncation=True)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
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
    args = parser.parse_args()
    mode = args.mode

    print(f'{mode} mode selected ..')

    # Assuming bias_direction is your pre-computed bias direction tensor
    # Load it here, for example:
    bias_direction = torch.load(f'/rds/general/user/bsk18/home/sentence-debiasing/subspaces/subspace_m-GPT2Model_c-gpt2_t-{mode}.pt')
    model = SentenceDebiasGPT2LMHeadModel('gpt2', bias_direction)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')


    output_file = f'/rds/general/user/bsk18/home/sentence-debiasing/generated_debias_texts_{mode}.jsonl'
    prompts = read_prompts('/rds/general/user/bsk18/home/sentence-debiasing/debias_prompts.jsonl')

    for idx, prompt in enumerate(prompts):
        generated_text = generate_text(prompt, model, tokenizer)
        data = {
            'prompt': prompt,
            'generated_text': generated_text
        }
        write_to_jsonl(output_file, data)
        print(f"{idx}/{len(prompts)} done ...")

    print("Debiasing completed!")
