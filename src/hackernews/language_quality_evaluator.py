from hackernews.UniEval.utils import convert_to_json
from hackernews.UniEval.metric.evaluator import get_evaluator
import nltk
nltk.download('punkt')

class Language_Evaluator:

    def __init__(self, prompt: str, job_desc: str, print_result=True):
        
        self.src = prompt
        self.context =prompt
        self.output = job_desc
        self.print = print_result

    def language_score(self):

        task = 'dialogue'

        # a list of dialogue histories
        # src_list = ['The job is located in Edinburgh Genome Foundry. The company, Edinburgh Genome Foundry, is seeking a qualified individual for the Senior Software Engineer position. The ideal candidate would be skilled in the following technologies: open-source. The remote work options for this job are currently unknown. Write a detailed job description based on this information.\n\n']
        src_list = [self.src]

        # a list of additional context that should be included into the generated response
        # context_list = ['The job is located in Edinburgh Genome Foundry. The company, Edinburgh Genome Foundry, is seeking a qualified individual for the Senior Software Engineer position. The ideal candidate would be skilled in the following technologies: open-source. The remote work options for this job are currently unknown. Write a detailed job description based on this information.\n\n']
        context_list = self.context

        # a list of model outputs to be evaluated
        # output_list = ['Sub enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise Enterprise']
        output_list = [self.output]
                    
        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=output_list, 
                            src_list=src_list, context_list=context_list)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, print_result=True)

        # return overall score
        return eval_scores[-1]['overall']
            
