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
from hackernews.language_quality_evaluator import Language_Evaluator


class Diversity_Evaluator:

    def __init__(self, target_male_pct: float=0.5, target_female_pct:float=0.5):

        self.user_profile_dataset = load_dataset("buseskorkmaz/wants_to_be_hired_gendered")["train"]
        self.hiring_dataset = load_dataset("buseskorkmaz/hiring_w_q_context", split='train')
        self.language_eval = Language_Evaluator()
        items = [row for row in self.hiring_dataset]

        # remove links <a> and </a> are special tokens
        def remove_links(text):
            clean_text = re.sub('<a.*?</a>', '', text)
            clean_text = clean_text.replace('<a href="', '')
            clean_text = clean_text.replace('www.', '')
            clean_text = clean_text.replace("</a>", '')
            clean_text = clean_text.replace('"', '')
            return clean_text

        self.text2embedding = {remove_links(item['text']): item['embedding'] for item in items}
        self.prompt2idx = {remove_links(items[idx]['prompt']): idx for idx in range(len(items))}
        self.idx2location = {idx: items[idx]['location'] for idx in range(len(items))}
        self.promtp2original = {remove_links(items[idx]['prompt']): items[idx]['text'] for idx in range(len(items))}

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

        # Tokenize and pad the text to a maximum length of 512 tokens
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')

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
    

    def calc_q_value(self, job_desc, prompt):

        if job_desc in self.text2embedding.keys():
            job_embedding = self.text2embedding[job_desc]
        else:
            job_embedding = self.encode_text(job_desc)
        
        idx = None
        if prompt in self.prompt2idx.keys():
            idx = self.prompt2idx[prompt]
            job_location = self.idx2location[idx]
            original = self.promtp2original[prompt]
            original_embedding = self.encode_text(original)
        else:
            raise NotImplementedError
        
        # overall language check for coherency
        language_scores = self.language_eval.language_score(prompt, job_desc)
        language_score = language_scores['overall'] 
        if language_score < 0.5:
            print("Poor English quality")
            language_value = -1000
        else:
            language_value = language_score * 100
        
        if cosine_similarity(np.array(job_embedding).reshape(1, -1), np.array(original_embedding).reshape(1, -1))[0][0] >= 0.98:
            language_value *= 0.6 

        # # fact check based on given prompt
        # groundedness = language_scores['groundedness']
        # factual_penalty = 0
        # if groundedness < 0.85:
        #     print('It is not grounded enough for given prompt')
        #     factual_penalty = (0.85 - groundedness) * -100

        k = 100
        locations = []
        main_locations = []
        genders = []
        filtered_user_profiles = self.user_profile_dataset.filter(lambda x: self.filter_candidates(x,job_location))
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

            # target_location_distribution = np.array(list(get_element_percentages(user_profile_dataset["Main_location"]).values()))
            # realized_location_distribution = np.array(list(get_element_percentages(filtered_user_profiles["Main_location"]).values()))
            target_location_distribution =  self.target_location_distribution
            realized_location_distribution = np.array([real_na_pct, real_eu_pct, real_asia_pct, real_remote_pct, real_australia_pct, real_africa_pct,
                                                    real_sa_pct, real_unknown_pct])

            print(target_location_distribution)
            print(realized_location_distribution)

            # Calculate the KL divergence -- too sensitive easily becoming inf
            # kl_divergence_gender = kl_div(target_gender_distribution, realized_gender_distribution).sum()
            # kl_divergence_location = kl_div(target_location_distribution, realized_location_distribution).sum()

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

            # print("The most similar profile cosine similarity:", similarity_matrix[idmax])
            # print("=="*35)
            # print(user_profile_dataset[int(idmax)]["text"])
            # print("=="*35)
            # print("Gender:", user_profile_dataset[int(idmax)]["Gender"])
            # print("=="*35)
            # print(job_desc["text"])
        else:
            print("no match")
            q_value = -1000
        
        # q_value += language_value + factual_penalty
        q_value += language_value
         
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
        
        return q_value

