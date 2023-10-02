import argparse
import pickle as pkl
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src'))
import dill
import re
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from transformers import BertTokenizer, BertModel
import torch
import pycountry

def alpha2_to_name(alpha_2):
    try:
        return pycountry.countries.get(alpha_2=alpha_2).name
    except LookupError:
        return None
    except AttributeError:
        return None

class Diversity_Evaluator:

    def __init__(self, evaluation_dataset, 
                #  target_male_pct: float=0.5, target_female_pct:float=0.5
                ):

        
        self.user_profile_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/candidates_w_main_location")
        self.hiring_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/job_descriptions_w_q_prompt_eng")
        items = [row for row in self.hiring_dataset]

        self.text2embedding = {item['description']: item['embedding'] for item in items}
        self.prompt2location = {items[idx]['prompt']: alpha2_to_name(items[idx]['job_country_code']) for idx in range(len(items))}
        self.promtp2original = {items[idx]['prompt']: items[idx]['description'] for idx in range(len(items))}

        # Load pre-trained model and tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # initialize target distributions
        self.calc_location_statistics()

        # target_male_pct = 0.5
        # target_female_pct  = 0.5 

        # self.target_gender_distribution= np.array([target_male_pct, target_female_pct])  # 50% male, 50% female
        

    def encode_text(self, job_desc):

        text = job_desc

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
    
    def calc_location_statistics(self):

        print("Calculating location diversity statistics of the dataset....")
        total_candidate = len(self.user_profile_dataset)
        target_na_pct = self.user_profile_dataset["Main_location"].count("North America") / total_candidate
        target_eu_pct = self.user_profile_dataset["Main_location"].count("Europe") / total_candidate
        target_asia_pct = self.user_profile_dataset["Main_location"].count("Asia") / total_candidate
        target_oceania_pct = self.user_profile_dataset["Main_location"].count("Oceania") / total_candidate
        target_africa_pct = self.user_profile_dataset["Main_location"].count("Africa") / total_candidate
        target_sa_pct = self.user_profile_dataset["Main_location"].count("South America") / total_candidate
        target_antarctica_pct =  self.user_profile_dataset["Main_location"].count("Antarctica") / total_candidate   

        self.target_location_distribution = np.array([target_na_pct, target_eu_pct, target_asia_pct, target_oceania_pct, target_africa_pct,
                                                    target_sa_pct, target_antarctica_pct])
        return
    

    def calc_q_value(self, job_desc, prompt):

        if job_desc in self.text2embedding.keys():
            job_embedding = self.text2embedding[job_desc]
        else:
            job_embedding = self.encode_text(job_desc)
        
        idx = None
        print(prompt)
        if prompt in self.prompt2location.keys():
            # idx = self.prompt2idx[prompt]
            # job_location = self.idx2location[idx]
            job_location = self.prompt2location[prompt]
            # original = self.promtp2original[prompt]
            # original_embedding = self.encode_text(original)
        else:
            raise NotImplementedError 

        k = 50
        locations = []
        main_locations = []
        genders = []
        # filtered_user_profiles = self.user_profile_dataset.filter(lambda x: self.filter_candidates(x,job_location))
        filtered_user_profiles = self.user_profile_dataset
        # print("init match:", len(filtered_user_profiles))
        if filtered_user_profiles:
            similarity_matrix = cosine_similarity(filtered_user_profiles["embedding"], np.array(job_embedding).reshape(1, -1)).flatten()
            idmax = similarity_matrix.argmax()
            ind = np.argsort(similarity_matrix)[::-1][:k]
            for idmax in ind:
                locations.append(filtered_user_profiles[int(idmax)]["Location"])
                main_locations.append(filtered_user_profiles[int(idmax)]["Main_location"])
                # genders.append(filtered_user_profiles[int(idmax)]["Gender"])
            
            # check if the information is correct
            assert len(main_locations) == np.shape(similarity_matrix)[0] or len(main_locations) == k

            # Define your distributions
            # These are probabilities and must sum to 1
            # real_male_pct = genders.count("Male")/len(genders)
            # real_female_pct = genders.count("Female")/len(genders)

            # target_gender_distribution = np.array([target_male_pct, target_female_pct])  # 50% male, 50% female
            # realized_gender_distribution = np.array([real_male_pct, real_female_pct])  # 30% male, 70% female
            # print(target_gender_distribution)
            # print(realized_gender_distribution)

            total_candidate = len(main_locations)
            real_na_pct = main_locations.count("North America") / total_candidate
            real_eu_pct = main_locations.count("Europe") / total_candidate
            real_asia_pct = main_locations.count("Asia") / total_candidate
            real_oceania_pct = main_locations.count("Oceania") / total_candidate
            real_africa_pct = main_locations.count("Africa") / total_candidate
            real_sa_pct = main_locations.count("South America") / total_candidate
            real_antarctica_pct =  main_locations.count("Antarctica") / total_candidate

            target_location_distribution =  self.target_location_distribution
            realized_location_distribution = np.array([real_na_pct, real_eu_pct, real_asia_pct, real_oceania_pct, real_africa_pct,
                                                    real_sa_pct, real_antarctica_pct])

            # print(target_location_distribution)
            print(realized_location_distribution)

            # Calculate the Wasserstein divergence -- always finite
            # wasserstein_distance_gender = wasserstein_distance(target_gender_distribution, realized_gender_distribution)
            wasserstein_distance_location = wasserstein_distance(target_location_distribution, realized_location_distribution)

            # print(f"The Wasserstein distance between the target gender and realized distributions is {wasserstein_distance_gender}")
            # print(f"The Wasserstein distance between the target location and realized distributions is {wasserstein_distance_location}")

            # if wasserstein_distance_gender == np.inf or wasserstein_distance_location == np.inf:
            #     distance = 10
            # else:
            #     distance =  wasserstein_distance_gender + wasserstein_distance_location

            if wasserstein_distance_location == np.inf:
                distance = 10
            else:
                distance = wasserstein_distance_location
            
            print(f"The total Wasserstein distance between the target and realized distributions is {distance}")
            q_value = distance * -100

            # idmax = similarity_matrix.argmax()
            # print("The most similar profile cosine similarity:", similarity_matrix[idmax])
            # print("=="*35)
            # print(self.user_profile_dataset[int(idmax)]["text"])
            # print("=="*35)
            # print("Gender:", user_profile_dataset[int(idmax)]["Gender"])
            # print("=="*35)
            # print(job_desc["text"])

            ind_selected = np.argsort(similarity_matrix)[::-1][:10]
            loc_selected = []
            # gender_selected = []
            for idmax in ind_selected:
                loc_selected.append(filtered_user_profiles[int(idmax)]["Main_location"])
                # gender_selected.append(filtered_user_profiles[int(idmax)]["Gender"])

            # selection rate of genders   
            # if real_female_pct > 0:
            #     sr_female = gender_selected.count("Female") / genders.count("Female")
            # else:
            #     sr_female = 0
            
            # if real_male_pct > 0:
            #     sr_male = gender_selected.count("Male") / genders.count("Male")
            # else:
            #     sr_male = 0

            # # impact ratio of genders
            # impact_r_female = sr_female/ max(sr_female, sr_male)
            # impact_r_male = sr_male / max(sr_female, sr_male)

            # print("IR F", impact_r_female, "IR M", impact_r_male)

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

            if real_oceania_pct > 0:
                sr_oceania = loc_selected.count("Oceania") / main_locations.count("Oceania") 
            else:
                sr_oceania = 0
            
            if real_africa_pct > 0:
                sr_africa = loc_selected.count("Africa") / main_locations.count("Africa")
            else:
                sr_africa = 0
            
            if real_sa_pct > 0:
                sr_sa = loc_selected.count("South America") / main_locations.count("South America") 
            else:
                sr_sa = 0
            
            if real_antarctica_pct > 0:
                sr_antarctica =  loc_selected.count("Antarctica") / main_locations.count("Antarctica")
            else:
                sr_antarctica = 0

            # impact ratio of locations
            impact_r_na = sr_na / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)
            impact_r_eu = sr_eu / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)
            impact_r_asia = sr_asia / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)
            impact_r_oceania = sr_oceania / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)
            impact_r_africa = sr_africa / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)
            impact_r_sa = sr_sa / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)
            impact_r_antarctica = sr_antarctica / max(sr_na, sr_eu, sr_asia, sr_oceania, sr_africa, sr_sa, sr_antarctica)

        else:
            print("no match")
            wasserstein_distance_gender = 1
            wasserstein_distance_location = 1
            distance =  wasserstein_distance_gender + wasserstein_distance_location
            q_value = -100

            real_na_pct = np.nan
            real_eu_pct = np.nan
            real_asia_pct = np.nan
            real_oceania_pct = np.nan
            real_africa_pct = np.nan
            real_sa_pct = np.nan
            real_antarctica_pct = np.nan

            # real_male_pct = np.nan
            # real_female_pct = np.nan

            # sr_female = np.nan
            # sr_male = np.nan

            # impact ratio of genders
            # impact_r_female = np.nan
            # impact_r_male = np.nan

            # selection rate of locations
            sr_na = np.nan
            sr_eu = np.nan
            sr_asia = np.nan
            sr_oceania = np.nan
            sr_africa = np.nan
            sr_sa = np.nan

            # impact ratio of locations
            impact_r_na = np.nan
            impact_r_eu = np.nan
            impact_r_asia = np.nan
            impact_r_oceania = np.nan
            impact_r_africa = np.nan
            impact_r_sa = np.nan
            impact_r_antarctica = np.nan
                 
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
        
        return {"evaluated_text": job_desc, 
                # "sr_female": sr_female, "sr_male": sr_male, "gender_distance": wasserstein_distance_gender, 
                "location_distance": wasserstein_distance_location,
                "sr_na": sr_na, "sr_eu": sr_eu, "sr_asia": sr_asia, "sr_oceania": sr_oceania, "sr_africa": sr_africa, "sr_sa":sr_sa,
                "sr_antarctica": sr_antarctica, "ir_na": impact_r_na, "ir_eu": impact_r_eu, "ir_asia": impact_r_asia,
                "ir_oceania": impact_r_oceania, "ir_africa": impact_r_africa, "ir_sa": impact_r_sa, "ir_antarctica": impact_r_antarctica,
                # "ir_female": impact_r_female, "ir_male": impact_r_male, 
                "q_val": q_value}

def extract_text(input_string, parent):
    if parent:
        pattern = r"(?<=parent:)(.*?)(?=comment:)"
    else:
        pattern = r"(?<=comment:)(.*)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    hiring_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/job_descriptions_w_q_prompt_eng")
    items = [row for row in hiring_dataset]
    prompt2idx = {items[idx]['prompt']: idx for idx in range(len(items))}

    with open(args.eval_file, 'rb') as f:
        d = dill.load(f)

    # print(d)
    # print("=="*50)
    # print(d['eval_dump'])
    # print([str(item[0]) for item in d['eval_dump']['results']])
    eval_indexes = []
    generated_texts = []
    for item in d['eval_dump']['results']:
        if sum(map(lambda x: x[2], item[1])) != -200:
            prompt  = extract_text(str(item[0]), parent=True)[0]
            generated_texts.append(extract_text(str(item[0]), parent=False)[0])
            eval_indexes.append(prompt2idx[prompt])    

    # print(generated_texts)
    # print(eval_indexes)
    
    evaluation_dataset = hiring_dataset.select(eval_indexes)
    evaluation_dataset = evaluation_dataset.add_column("generated_text", generated_texts)

    evaluator = Diversity_Evaluator(evaluation_dataset=evaluation_dataset)
    generated_evaluation_dataset = evaluation_dataset.map(lambda x: evaluator.calc_q_value(x["generated_text"], x["prompt"]))
    # original_evaluation_dataset = evaluation_dataset.map(lambda x: evaluator.calc_q_value(x["description"], x["prompt"]))
    
    generated_evaluation_dataset.save_to_disk(f"workable/{args.save_path}_generated")
    # original_evaluation_dataset.save_to_disk(f"workable/{args.save_path}_original")
    print(generated_evaluation_dataset[1])
    # # print(original_evaluation_dataset[1])
