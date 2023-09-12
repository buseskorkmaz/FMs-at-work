from datasets import load_from_disk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
random.seed(42)
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

    def __init__(self, 
                #  target_male_pct: float=0.5, 
                # target_female_pct:float=0.5,
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
        # genders = []
        # target_male_pct = 0.5
        # target_female_pct  = 0.5 
        # filtered_user_profiles = user_profile_dataset.filter(lambda x: filter_candidates(x,job_desc["job_country_code"]))
        filtered_user_profiles = self.user_profile_dataset
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
    
            
            # print(f"The total Wasserstein distance between the target and realized distributions is {distance}")
            q_value = distance * -100

            # print("The most similar profile cosine similarity:", similarity_matrix[idmax])
            # print("=="*35)
            # print(user_profile_dataset[int(idmax)]["profile_description"])
            # print("=="*35)
            # print(job_desc["description"])
        else:
            print("no match")
            q_value = -100
        
        # q_value += language_value

        print("Job desc:", job_desc, "\n") 
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
        # print("--"*50, "\n\n")  

        return q_value

