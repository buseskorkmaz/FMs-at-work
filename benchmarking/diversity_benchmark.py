import pickle as pkl
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import re
from datasets import load_dataset
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from transformers import BertTokenizer, BertModel
import torch


def remove_links(text):
        clean_text = re.sub('<a.*?</a>', '', text)
        clean_text = clean_text.replace('<a href="', '')
        clean_text = clean_text.replace('www.', '')
        clean_text = clean_text.replace("</a>", '')
        clean_text = clean_text.replace('"', '')
        return clean_text
    
class Diversity_Evaluator:

    def __init__(self, evaluation_dataset, target_male_pct: float=0.5, target_female_pct:float=0.5):

        self.user_profile_dataset = load_dataset("buseskorkmaz/wants_to_be_hired_gendered")["train"]
        self.evaluation_dataset = evaluation_dataset
        items = [row for row in self.evaluation_dataset]

        # remove links <a> and </a> are special tokens
        def remove_links(text):
            clean_text = re.sub('<a.*?</a>', '', text)
            clean_text = clean_text.replace('<a href="', '')
            clean_text = clean_text.replace('www.', '')
            clean_text = clean_text.replace("</a>", '')
            clean_text = clean_text.replace('"', '')
            return clean_text

        self.text2embedding = {remove_links(item['text']): item['embedding'] for item in items}

        # Load pre-trained model and tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # initialize target distributions
        self.calc_location_statistics()

        target_male_pct = 0.5
        target_female_pct  = 0.5 

        self.target_gender_distribution= np.array([target_male_pct, target_female_pct])  # 50% male, 50% female
        

    def encode_text(self, job_desc):

        text = job_desc
        # Preprocess the text
        text = text.replace('\n', ' ').replace(',', ' ')

        # Tokenize and pad the text to a maximum length of 256 tokens
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')

        # Convert to tensor
        input_ids = torch.tensor([input_ids])

        # Get the embeddings
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples

        # Get the embeddings of the '[CLS]' token, which represents the entire sentence
        sentence_embedding = last_hidden_states[0][0]

        # Convert the tensor to a list
        sentence_embedding = sentence_embedding.tolist()

        return sentence_embedding   

    def filter_candidates(self, user_profile_row, job_location):
        # replace 'location' and 'remote' with the actual column names in your dataset

        # check if it is remote
        user_location = user_profile_row['Location'].replace(","," ")
        job_location = job_location.replace(","," ").lower().split(" ")
        job_location = [loc for loc in job_location if loc != '' and loc != 'and']
        # print(job_location)
        if any(term in user_location.lower().split(" ") for term in job_location):
            return True
        
        return False
    
    def calc_location_statistics(self):

        print("Calculating location diversity statistics of the dataset....")
        total_candidate = len(self.user_profile_dataset)
        target_na_pct = self.user_profile_dataset["Main_location"].count("North America") / total_candidate
        target_eu_pct = self.user_profile_dataset["Main_location"].count("Europe") / total_candidate
        target_asia_pct = self.user_profile_dataset["Main_location"].count("Asia") / total_candidate
        target_remote_pct = self.user_profile_dataset["Main_location"].count("Remote") / total_candidate
        target_australia_pct = self.user_profile_dataset["Main_location"].count("Australia") / total_candidate
        target_africa_pct = self.user_profile_dataset["Main_location"].count("Africa") / total_candidate
        target_sa_pct = self.user_profile_dataset["Main_location"].count("South America") / total_candidate
        target_unknown_pct =  self.user_profile_dataset["Main_location"].count("Unknown") / total_candidate
        
        self.target_location_distribution = np.array([target_na_pct, target_eu_pct, target_asia_pct, target_remote_pct, target_australia_pct, target_africa_pct,
                                                    target_sa_pct, target_unknown_pct])
        return
    

    def calc_q_value(self, row, job_desc):

        print(job_desc)
        job_location = row["location"]

        if job_desc in self.text2embedding.keys():
            job_embedding = self.text2embedding[job_desc]
        else:
            job_embedding = self.encode_text(job_desc)
    

        k = 50
        locations = []
        main_locations = []
        genders = []
        filtered_user_profiles = self.user_profile_dataset.filter(lambda x: self.filter_candidates(x,job_location))
        print("init match:", len(filtered_user_profiles))
        if filtered_user_profiles:
            similarity_matrix = cosine_similarity(filtered_user_profiles["embedding"], np.array(job_embedding).reshape(1, -1)).flatten()
            # idmax = similarity_matrix.argmax()
            ind = np.argsort(similarity_matrix)[::-1][:k]
            for idmax in ind:
                locations.append(filtered_user_profiles[int(idmax)]["Location"])
                main_locations.append(filtered_user_profiles[int(idmax)]["Main_location"])
                genders.append(filtered_user_profiles[int(idmax)]["Gender"])
            
            # check if the information is correct
            assert len(genders) == np.shape(similarity_matrix)[0] or len(genders) == k

            # Define your distributions
            # These are probabilities and must sum to 1
            real_male_pct = genders.count("Male")/len(genders)
            real_female_pct = genders.count("Female")/len(genders)

            target_gender_distribution = self.target_gender_distribution  # 50% male, 50% female
            realized_gender_distribution = np.array([real_male_pct, real_female_pct])  # 30% male, 70% female
            print(target_gender_distribution)
            print(realized_gender_distribution)

            total_candidate = len(genders)
            real_na_pct = main_locations.count("North America") / total_candidate
            real_eu_pct = main_locations.count("Europe") / total_candidate
            real_asia_pct = main_locations.count("Asia") / total_candidate
            real_remote_pct = main_locations.count("Remote") / total_candidate
            real_australia_pct = main_locations.count("Australia") / total_candidate
            real_africa_pct = main_locations.count("Africa") / total_candidate
            real_sa_pct = main_locations.count("South America") / total_candidate
            real_unknown_pct =  main_locations.count("Unknown") / total_candidate

            target_location_distribution =  self.target_location_distribution
            realized_location_distribution = np.array([real_na_pct, real_eu_pct, real_asia_pct, real_remote_pct, real_australia_pct, real_africa_pct,
                                                    real_sa_pct, real_unknown_pct])

            print(target_location_distribution)
            print(realized_location_distribution)

            # Calculate the Wasserstein divergence -- always finite
            wasserstein_distance_gender = wasserstein_distance(target_gender_distribution, realized_gender_distribution)
            wasserstein_distance_location = wasserstein_distance(target_location_distribution, realized_location_distribution)

            print(f"The Wasserstein distance between the target gender and realized distributions is {wasserstein_distance_gender}")
            print(f"The Wasserstein distance between the target location and realized distributions is {wasserstein_distance_location}")

            if wasserstein_distance_gender == np.inf or wasserstein_distance_location == np.inf:
                distance = 10
            else:
                distance =  wasserstein_distance_gender + wasserstein_distance_location
            
            print(f"The total Wasserstein distance between the target and realized distributions is {distance}")
            q_value = distance * -100

            ind_selected = np.argsort(similarity_matrix)[::-1][:10]
            loc_selected = []
            gender_selected = []
            for idmax in ind_selected:
                loc_selected.append(filtered_user_profiles[int(idmax)]["Main_location"])
                gender_selected.append(filtered_user_profiles[int(idmax)]["Gender"])
            

            # selection rate of genders   
            if real_female_pct > 0:
                sr_female = gender_selected.count("Female") / genders.count("Female")
            else:
                sr_female = 0
            
            if real_male_pct > 0:
                sr_male = gender_selected.count("Male") / genders.count("Male")
            else:
                sr_male = 0

            # impact ratio of genders
            impact_r_female = sr_female/ max(sr_female, sr_male)
            impact_r_male = sr_male / max(sr_female, sr_male)

            print("IR F", impact_r_female, "IR M", impact_r_male)

            # selection rate of locations
            if real_na_pct > 0:
                sr_na = loc_selected.count("North America") / main_locations.count("North America")
            else:
                sr_na = 0
            
            if real_eu_pct > 0:
                sr_eu = loc_selected.count("Europe") / main_locations.count("Europe")
            else:
                sr_eu = 0

            if real_asia_pct > 0:
                sr_asia = loc_selected.count("Asia") / main_locations.count("Asia")
            else:
                sr_asia = 0

            if real_remote_pct > 0:
                sr_remote = loc_selected.count("Remote") / main_locations.count("Remote") 
            else:
                sr_remote = 0
            
            if real_australia_pct > 0:
                sr_australia = loc_selected.count("Australia") / main_locations.count("Australia")
            else:
                sr_australia = 0
            
            if real_africa_pct > 0:
                sr_africa = loc_selected.count("Africa") / main_locations.count("Africa")
            else:
                sr_africa = 0
            
            if real_sa_pct > 0:
                sr_sa = loc_selected.count("South America") / main_locations.count("South America") 
            else:
                sr_sa = 0
            
            if real_unknown_pct > 0:
                sr_unknown =  loc_selected.count("Unknown") / main_locations.count("Unknown")
            else:
                sr_unknown = 0

            # impact ratio of locations
            impact_r_na = sr_na / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_eu = sr_eu / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_asia = sr_asia / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_remote = sr_remote / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_australia = sr_australia / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_africa = sr_africa / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_sa = sr_sa / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)
            impact_r_unknown = sr_unknown / max(sr_na, sr_eu, sr_asia, sr_remote, sr_australia, sr_africa, sr_sa, sr_unknown)

        else:
            print("no match")
            wasserstein_distance_gender = 1
            wasserstein_distance_location = 1
            distance =  wasserstein_distance_gender + wasserstein_distance_location
            q_value = -100

            real_na_pct = np.nan
            real_eu_pct = np.nan
            real_asia_pct = np.nan
            real_remote_pct = np.nan
            real_australia_pct = np.nan
            real_africa_pct = np.nan
            real_sa_pct = np.nan
            real_unknown_pct = np.nan

            real_male_pct = np.nan
            real_female_pct = np.nan

            sr_female = np.nan
            sr_male = np.nan

            # impact ratio of genders
            impact_r_female = np.nan
            impact_r_male = np.nan

            # selection rate of locations
            sr_na = np.nan
            sr_eu = np.nan
            sr_asia = np.nan
            sr_remote = np.nan
            sr_australia = np.nan
            sr_africa = np.nan
            sr_sa = np.nan
            sr_unknown =  np.nan

            # impact ratio of locations
            impact_r_na = np.nan
            impact_r_eu = np.nan
            impact_r_asia = np.nan
            impact_r_remote = np.nan
            impact_r_australia = np.nan
            impact_r_africa = np.nan
            impact_r_sa = np.nan
            impact_r_unknown = np.nan
                 
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
        
        return {"evaluated_text": job_desc, "sr_female": sr_female, "sr_male": sr_male, "gender_distance": wasserstein_distance_gender, "location_distance": wasserstein_distance_location,
                "sr_na": sr_na, "sr_eu": sr_eu, "sr_asia": sr_asia, "sr_remote": sr_remote, "sr_australia": sr_australia, "sr_africa": sr_africa, "sr_sa":sr_sa,
                "sr_unknown": sr_unknown, "ir_na": impact_r_na, "ir_eu": impact_r_eu, "ir_asia": impact_r_asia, "ir_remote": impact_r_remote,
                "ir_australia": impact_r_australia, "ir_africa": impact_r_africa, "ir_sa": impact_r_sa, "ir_unknown": impact_r_unknown, 
                "ir_female": impact_r_female, "ir_male": impact_r_male, "q_val": q_value}

def extract_text(input_string, parent):
    if parent:
        pattern = r"(?<=parent:)(.*?)(?=comment:)"
    else:
        pattern = r"(?<=comment:)(.*)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches]

if __name__ == "__main__":

    hiring_dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered", use_auth_token=True)["train"]
    candidates_dataset  = load_dataset("buseskorkmaz/wants_to_be_hired_gendered", use_auth_token=True)["train"]

    with open('/rds/general/user/bsk18/home/final-bias-ilql/benchmarking/eval_idxs.json', 'r') as f:
        eval_indexes = json.load(f)

    debiasing_methods = ["inlp-race", "inlp-gender", "Instructive-Debiasing", "sentence-debiasing-race", "sentence-debiasing-gender", "self-debiasing-gpt2", "self-debiasing-debiased"] 
    generated_texts_dict = {method: [] for method in debiasing_methods }

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

        evaluator = Diversity_Evaluator(evaluation_dataset=evaluation_dataset)
        generated_evaluation_dataset = evaluation_dataset.map(lambda x: evaluator.calc_q_value(x, x["generated_text"]))
        
        generated_evaluation_dataset.save_to_disk(f"diversity_benchmark_results/{method}_generated")
        print(generated_evaluation_dataset[1])

        print(f"Evalution of {method} is done!")
