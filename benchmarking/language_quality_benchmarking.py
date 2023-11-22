from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from datasets import load_dataset
import json
import typing as T

class LanguageEvaluator:

    def eval(
        self,
        src_list: T.List,
        output_list: T.List,
        context_list: T.List=[],
    )-> T.Dict:   

        task = 'dialogue'
        if context_list == []:
            context_list = src_list
    
        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=output_list, 
                            src_list=src_list, context_list=context_list)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, print_result=True)

        return eval_scores
    
if __name__ == "__main__":

    hiring_dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered", use_auth_token=True)["train"]

    with open('/rds/general/user/bsk18/home/final-bias-ilql/benchmarking/eval_idxs.json', 'r') as f:
        eval_indexes = json.load(f)

    debiasing_methods = ["inlp-race", "inlp-gender", "Instructive-Debiasing", "sentence-debiasing-race", "sentence-debiasing-gender", "self-debiasing-gpt2", "self-debiasing-debiased"] 
    generated_texts_dict = {method: [] for method in debiasing_methods }
    evaluation_dict = {}
    for method in debiasing_methods:

        print(f'Evaluating {method} ...')

        if method == "inlp-race":
            file_path = 'outputs/inlp/generated_debias_texts_race.jsonl'    
            generated_text_key = 'generated_text' 
        
        elif method == "inlp-gender":
            file_path = 'outputs/inlp/generated_debias_texts_gender.jsonl'    
            generated_text_key = 'generated_text' 
    
        elif method == "sentence-debiasing-race":
            file_path = 'outputs/sentence-debiasing/generated_debias_texts_race.jsonl'    
            generated_text_key = 'generated_text' 

        elif method == "sentence-debiasing-gender":
            file_path = 'outputs/sentence-debiasing/generated_debias_texts_gender.jsonl'    
            generated_text_key = 'generated_text' 
        
        elif method == "Instructive-Debiasing":
            file_path = 'outputs/Instructive-Debiasing/Continuations-text-davinci-003-debiased.txt'    
            generated_text_key = 'Continuation' 

        elif method == "self-debiasing-gpt2":
            file_path = 'outputs/self-debiasing/prompted_generations_gpt2-large_default.txt'
            generated_text_key = 'continuations'
        
        elif method == "self-debiasing-debiased":
            file_path = 'outputs/self-debiasing/prompted_generations_gpt2-large_debiased.txt'
            generated_text_key = 'continuations'

        
        # List to hold the extracted generated_text values
        generated_texts = []
        file_path = '/rds/general/user/bsk18/home/final-bias-ilql/benchmarking/' + file_path    
        # Open the file and read line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Parse the JSON data from each line
                json_data = json.loads(line.strip())

                # Extract the 'generated_text' field
                generated_text = json_data.get(generated_text_key, None)

                if method in ["self-debiasing-gpt2","self-debiasing-debiased"]:
                    generated_text = generated_text[0]['text']

                if generated_text is not None:
                    generated_texts.append(generated_text)
    
        generated_texts_dict[method] = generated_texts
    
        evaluation_dataset = hiring_dataset.select(eval_indexes)
        evaluation_dataset = evaluation_dataset.add_column("generated_text", generated_texts)
        
        evaluator = LanguageEvaluator()
        language_quality_scores= evaluator.eval(src_list=evaluation_dataset['prompt'] ,output_list=evaluation_dataset['generated_text'])
        evaluation_dict[method] = language_quality_scores

    # Specify the file path for the JSON file
    output_file_path = '/rds/general/user/bsk18/home/final-bias-ilql/benchmarking/language_quality_results/self_debiasing_language_quality_scores.json'

    # Save the evaluation_dict to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
