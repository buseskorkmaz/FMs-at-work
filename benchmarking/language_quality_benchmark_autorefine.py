from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from datasets import load_dataset
import json
import typing as T
import pandas as pd

class LanguageEvaluator:

    def eval(
        self,
        src_list: T.List,
        output_list: T.List,
        context_list: T.List=[],
    )-> T.Dict:   

        # print(src_list)   

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

    debiasing_methods = ["original", "generated"] #later add self-debiasing when it is done
    evaluation_dict = {}
    for method in debiasing_methods:

        print(f'Evaluating {method} ...')
        file_path = f'analysis/input_files/hackernews_{method}_eval_df.csv'
        
        # List to hold the extracted generated_text values
        generated_texts = []
        file_path = '/rds/general/user/bsk18/home/final-bias-ilql/' + file_path    
        # Open the file and read line by line

        if method == "original":
            key = "text"
        else:
            key = "generated_text"

        generated_texts = pd.read_csv(file_path)[key].values
        generated_texts = [str(text) if not pd.isna(text) else "" for text in generated_texts]

        evaluation_dataset = hiring_dataset.select(eval_indexes)
        evaluation_dataset = evaluation_dataset.add_column("generated_text", generated_texts)
        
        evaluator = LanguageEvaluator()
        language_quality_scores= evaluator.eval(src_list=evaluation_dataset['prompt'] ,output_list=evaluation_dataset['generated_text'])
        evaluation_dict[method] = language_quality_scores

        # Specify the file path for the JSON file
        output_file_path = f'/rds/general/user/bsk18/home/final-bias-ilql/benchmarking/results/original_{method}_language_quality_scores.json'

        # Save the evaluation_dict to a JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
