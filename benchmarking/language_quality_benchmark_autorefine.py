from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from datasets import load_dataset, load_from_disk
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

    with open('$HOME/FMs-at-work/benchmarking/language_quality_results/eval_indexes.json', 'r') as f:
        eval_indexes = json.load(f)

    # debiasing_methods = ["original", "generated"] #later add self-debiasing when it is done
    debiasing_methods = ["generated"] #later add self-debiasing when it is done

    evaluation_dict = {}
    for method in debiasing_methods:

        print(f'Evaluating {method} ...')
        file_path = f'analysis/input_files/hackernews_{method}_eval_df.csv'
        
        # List to hold the extracted generated_text values
        generated_texts = []
        file_path = "$HOME/FMs-at-work/benchmarking/diversity_benchmark_results/openllamav2_generated"   
        # Open the file and read line by line

        if method == "original":
            key = "text"
        else:
            key = "generated_text"

        generated_texts = load_from_disk(file_path)
        generated_texts = generated_texts.to_pandas()
        generated_texts = [str(text) if not pd.isna(text) else "" for text in generated_texts]
        print(len(generated_texts))
        print(len(eval_indexes))
        print(len(set(eval_indexes)))
        # evaluation_dataset = hiring_dataset.select(eval_indexes)
        # evaluation_dataset = evaluation_dataset.add_column("generated_text", generated_texts)
        
        # evaluator = LanguageEvaluator()
        # language_quality_scores= evaluator.eval(src_list=evaluation_dataset['prompt'] ,output_list=evaluation_dataset['generated_text'])
        # evaluation_dict[method] = language_quality_scores

        # # Specify the file path for the JSON file
        # output_file_path = f'$HOME/FMs-at-work/benchmarking/language_quality_results/openllama_{method}_language_quality_scores.json'

        # # Save the evaluation_dict to a JSON file
        # with open(output_file_path, 'w', encoding='utf-8') as f:
        #     json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
