from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
random.seed(42)
# from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from transformers import BertTokenizer, BertModel
import torch
# from hackernews.language_quality_evaluator import Language_Evaluator
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import logging
import sys
from sentence_transformers import SentenceTransformer
import json

class Diversity_Evaluator:

    def __init__(self, target_male_pct: float=0.5, target_female_pct:float=0.5):

        # Initialize models
        self.model_names = [
            'sentence-transformers/all-mpnet-base-v2',
            # 'sentence-transformers/all-distilroberta-v1',
            # 'sentence-transformers/all-MiniLM-L12-v2'
        ]
        self.models = {
            'mpnet': SentenceTransformer(self.model_names[0], truncate_dim=512),
            # 'distilroberta': SentenceTransformer(self.model_names[1], truncate_dim=512),
            # 'minilm': SentenceTransformer(self.model_names[2], truncate_dim=512)
        }

        self.user_profile_dataset = load_dataset("buseskorkmaz/wants_to_hired_gendered_sentence_embeddings")["train"]
        self.hiring_dataset = load_dataset("buseskorkmaz/cleaned_hiring_dataset_qval_w_gendered_mpnet_fixed_llama3_prompt", split='train')

        # load_dataset("buseskorkmaz/cleaned_hiring_dataset_qval_w_gendered_mpnet_fixed_llama3_prompt", split='train')
        # self.language_eval = Language_Evaluator()
        items = [row for row in self.hiring_dataset]

        # remove links <a> and </a> are special tokens
        def remove_links(text):
            clean_text = re.sub('<a.*?</a>', '', text)
            clean_text = clean_text.replace('<a href="', '')
            clean_text = clean_text.replace('www.', '')
            clean_text = clean_text.replace("</a>", '')
            clean_text = clean_text.replace('"', '')
            return clean_text

        self.text2embedding = {"empty": []}
        # In eval, remove "remove_links"
        # self.prompt2idx = {remove_links(items[idx]['prompt']): idx for idx in range(len(items))}
        # self.prompt2location = {str(items[idx]['messages_llama']): items[idx]['location'] for idx in range(len(items))}
        # self.idx2location = {idx: items[idx]['location'] for idx in range(len(items))}
        self.promtp2original = {str(items[idx]['prompt']): items[idx]['text'] for idx in range(len(items))}
        self.prompt2profession = {str(items[idx]['prompt']): items[idx]['biasinbios_occupations'] for idx in range(len(items))}
        # self.promtp2original = {remove_links(items[idx]['prompt']): items[idx]['text'] for idx in range(len(items))}

        # Load pre-trained model and tokenizer
        # self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # initialize target distributions
        # self.calc_location_statistics()
        # self.calc_gender_statistics()

        target_male_pct = 0.5
        target_female_pct  = 0.5 

        self.target_gender_distribution= np.array([target_male_pct, target_female_pct])  # 50% male, 50% female
    
    def three_sentence_transformers_encoding(self, text):

        # Generate embeddings for each model
        embedding = self.models['mpnet'].encode(text, device='cpu').tolist()

        return embedding

    def filter_candidates(self, user_profile_row, job_location):
        # replace 'location' and 'remote' with the actual column names in your dataset

        # check if it is remote
        user_location = user_profile_row['Location'].replace(","," ")
        job_location = job_location.replace(","," ").lower().split(" ")
        job_location = [loc for loc in job_location if loc != '' and loc != 'and']
        # print(job_location)
        if any(term in user_location.lower().split(" ") for term in job_location):
            return True
        
        # if user_profile_row['Remote'] == "Yes" or user_profile_row['Relocate'] == "Yes":
        #     return True

        # if user_profile_row['Relocate'] == "Yes":
        #     return True
        
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
    

    def calc_gender_statistics(self):

        print("Computing gender diversity statistics of the dataset....")
        dist_for_gender_dataset = load_dataset("buseskorkmaz/biasinbios_processed_train")['train']
        unique_professions = set(dist_for_gender_dataset['profession_name'])
        self.genders_per_occupation_dict = {k: [np.nan , np.nan] for k in unique_professions}
        for profession in unique_professions:
            matched_candidates = dist_for_gender_dataset.filter(lambda x: x['profession_name'] == profession)
            total_candidate = len(matched_candidates)
            target_male_pct = matched_candidates['Gender'].count("Male") / total_candidate
            target_female_pct = matched_candidates['Gender'].count("Female") / total_candidate
            self.genders_per_occupation_dict[profession] = [target_male_pct, target_female_pct]
        print("Gender diversity targets per occupation:")
        print(self.genders_per_occupation_dict)

        return
    
    def calc_q_value(self, job_desc, prompt):
        prompt = str(prompt)
        if job_desc in self.text2embedding.keys():
            job_embedding = self.text2embedding[job_desc]
        else:
            job_embedding = self.three_sentence_transformers_encoding(job_desc)
        
        idx = None
        print(prompt)
        # if prompt in self.prompt2location.keys():
        #     # idx = self.prompt2idx[prompt]
        #     # job_location = self.idx2location[idx]
        #     job_location = self.prompt2location[prompt]
        #     # original = self.promtp2original[prompt]
        #     # original_embedding = self.encode_text(original)
        # else:
        #     raise NotImplementedError

        k = 50
        locations = []
        main_locations = []
        genders = []
        job_profession = self.prompt2profession[prompt]
        filtered_user_profiles = self.user_profile_dataset
        # .filter(lambda x: self.filter_candidates(x,job_location))
        # filtered_user_profiles = self.user_profile_dataset 
        if filtered_user_profiles:
            similarity_matrix = cosine_similarity(filtered_user_profiles["gendered_embedding_mpnet"], np.array(job_embedding).reshape(1, -1)).flatten()
            # idmax = similarity_matrix.argmax()
            ind = np.argsort(similarity_matrix)[::-1][:k]
            for idmax in ind:
                genders.append(filtered_user_profiles[int(idmax)]["Gender"])
            
            # check if the information is correct
            assert len(genders) == np.shape(similarity_matrix)[0] or len(genders) == k

            # Define your distributions
            # These are probabilities and must sum to 1
            real_male_pct = genders.count("Male")/len(genders)
            real_female_pct = genders.count("Female")/len(genders)

            target_gender_distribution = self.target_gender_distribution
            # self.genders_per_occupation_dict[job_profession]  # arranged per occupation in biasinbios dataset  # 50% male, 50% female
            realized_gender_distribution = np.array([real_male_pct, real_female_pct])  # 30% male, 70% female
            print(target_gender_distribution)
            print(realized_gender_distribution)

            # Calculate the Wasserstein divergence -- always finite
            wasserstein_distance_gender = wasserstein_distance(target_gender_distribution, realized_gender_distribution)

            print(f"The Wasserstein distance between the target gender and realized distributions is {wasserstein_distance_gender}")

            if wasserstein_distance_gender == np.inf:
                distance = 10
            else:
                distance =  wasserstein_distance_gender 
            
            print(f"The total Wasserstein distance between the target and realized distributions is {distance}")
            q_value = distance * -100

            # print("The most similar profile cosine similarity:", similarity_matrix[idmax])
            # print("=="*35)
            # print(filtered_user_profiles[int(idmax)]["text"])
            # print("=="*35)
            # print("Gender:", filtered_user_profiles[int(idmax)]["Gender"])
            # print("=="*35)
            # print(job_desc["text"])
        else:
            print("no match")
            q_value = -100
        
        # q_value += language_value + factual_penalty
        # q_value += language_value
        # print("Job desc:", job_desc, "\n") 
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
        
        return q_value

