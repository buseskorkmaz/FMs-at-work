import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src'))
import re
from datasets import load_dataset, load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from transformers import BertTokenizer, BertModel
import torch
import logging
from sentence_transformers import SentenceTransformer

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def remove_links(text):
    clean_text = re.sub('<a.*?</a>', '', text)
    clean_text = clean_text.replace('<a href="', '')
    clean_text = clean_text.replace('www.', '')
    clean_text = clean_text.replace("</a>", '')
    clean_text = clean_text.replace('"', '')
    return clean_text

class Diversity_Evaluator:

    def __init__(self, evaluation_dataset, target_male_pct: float=0.5, target_female_pct:float=0.5):

        self.user_profile_dataset = load_dataset("buseskorkmaz/wants_to_hired_gendered_sentence_embeddings", revision='ec4030f3b32e330178c05e136aa4d1c33bb355da')["train"]
        logging.info(f"dataset {self.user_profile_dataset}")
        logging.info(f"user profile dataset after {len(self.user_profile_dataset)}")
        
        self.evaluation_dataset = evaluation_dataset
        items = [row for row in self.evaluation_dataset]
        print(items[0])

        self.text2embedding = {item['cleaned_text']: item['embedding'] for item in items}
        self.text2embedding_mpnet = {item['cleaned_text']: item['cleaned_embedding_mpnet'] for item in items}
        self.text2embedding_distilroberta = {item['cleaned_text']: item['cleaned_embedding_distilroberta'] for item in items}
        self.text2embedding_minilm = {item['cleaned_text']: item['cleaned_embedding_minilm'] for item in items}
    
        # Load pre-trained BERT model and tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load SentenceTransformer models
        self.models = {
            'mpnet': SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
            'distilroberta': SentenceTransformer('sentence-transformers/all-distilroberta-v1'),
            'minilm': SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        }

        # initialize target distributions
        # self.calc_gender_statistics()
        self.target_gender_distribution = [target_male_pct, target_female_pct]

        logging.info("Diversity_Evaluator initialized.")

    def encode_text(self, job_desc, model_name=None):

        text = job_desc
        # Preprocess the text
        text = text.replace('\n', ' ').replace(',', ' ')

        if model_name is None:
            # Use BERT
            input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')
            input_ids = torch.tensor([input_ids])
            with torch.no_grad():
                last_hidden_states = self.model(input_ids)[0]
            sentence_embedding = last_hidden_states[0][0].tolist()
        else:
            # Use SentenceTransformer model
            sentence_embedding = self.models[model_name].encode(text).tolist()

        return sentence_embedding   

    def filter_candidates(self, user_profile_row, job_location):
        user_location = user_profile_row['Location'].replace(","," ")
        job_location = job_location.replace(","," ").lower().split(" ")
        job_location = [loc for loc in job_location if loc != '' and loc != 'and']
        if any(term in user_location.lower().split(" ") for term in job_location):
            return True
        return False

    # def calc_gender_statistics(self):
    #     print("Computing gender diversity statistics of the dataset....")
    #     unique_professions = set(self.user_profile_dataset['profession_name'])
    #     self.genders_per_occupation_dict = {k: [np.nan , np.nan] for k in unique_professions}
    #     for profession in unique_professions:
    #         matched_candidates = self.user_profile_dataset.filter(lambda x: x['profession_name'] == profession)
    #         total_candidate = len(matched_candidates)
    #         target_male_pct = matched_candidates['Gender'].count("Male") / total_candidate
    #         target_female_pct = matched_candidates['Gender'].count("Female") / total_candidate
    #         self.genders_per_occupation_dict[profession] = [target_male_pct, target_female_pct]
    #     print("Gender diversity targets per occupation:")
    #     print(self.genders_per_occupation_dict)

    #     return
    
    def calc_q_value(self, row, job_desc, embedding_type='embedding'):
        logging.info(f"Calculating Q-value for job description: {job_desc}")
        print(job_desc)

        if embedding_type == 'embedding':
            text2embedding = self.text2embedding
        elif embedding_type == 'mpnet':
            text2embedding = self.text2embedding_mpnet
        elif embedding_type == 'distilroberta':
            text2embedding = self.text2embedding_distilroberta
        elif embedding_type == 'minilm':
            text2embedding = self.text2embedding_minilm
        elif embedding_type.startswith('gendered'):
            text2embedding = getattr(self, f"text2{embedding_type.replace('gendered_', '')}")

        if job_desc in text2embedding.keys():
            job_embedding = text2embedding[job_desc]
        else:
            print("It requires new embedding...")
            job_embedding = self.encode_text(job_desc, model_name=embedding_type if embedding_type != 'embedding' else None)
        
        k = 50
        job_profession = row["biasinbios_occupations"]
        professions = []
        genders = []
        if not embedding_type.startswith("gendered"):
            user_profile_col_name = f"embedding_{embedding_type}" if embedding_type != "embedding" else "embedding"
        else:
            user_profile_col_name = f"gendered_{embedding_type.replace('gendered_', '')}"

        filtered_user_profiles = self.user_profile_dataset
        print("init match:", len(filtered_user_profiles))
        if filtered_user_profiles:
            similarity_matrix = cosine_similarity(filtered_user_profiles[user_profile_col_name], np.array(job_embedding).reshape(1, -1)).flatten()
            ind = np.argsort(similarity_matrix)[::-1][:k]
            for idmax in ind:
                genders.append(filtered_user_profiles[int(idmax)]["Gender"])
                # professions.append(filtered_user_profiles[int(idmax)]["profession_name"])
            
            assert len(genders) == np.shape(similarity_matrix)[0] or len(genders) == k

            real_male_pct = genders.count("Male")/len(genders)
            real_female_pct = genders.count("Female")/len(genders)

            target_gender_distribution = self.target_gender_distribution
            # self.genders_per_occupation_dict[job_profession]
            realized_gender_distribution = np.array([real_male_pct, real_female_pct])
            print(target_gender_distribution)
            print(realized_gender_distribution)

            # male_profession_percentages = {}
            # female_profession_percentages = {}
            # male_candidates = [prof for prof, gender in zip(professions, genders) if gender == "Male"]
            # female_candidates = [prof for prof, gender in zip(professions, genders) if gender == "Female"]
            # total_male_candidates = len(male_candidates)
            # total_female_candidates = len(female_candidates)
            
            # for profession in set(male_candidates):
            #     male_profession_percentages[profession] = male_candidates.count(profession) / total_male_candidates
            
            # for profession in set(female_candidates):
            #     female_profession_percentages[profession] = female_candidates.count(profession) / total_female_candidates

            wasserstein_distance_gender = wasserstein_distance(target_gender_distribution, realized_gender_distribution)

            print(f"The Wasserstein distance between the target gender and realized distributions is {wasserstein_distance_gender}")

            if wasserstein_distance_gender == np.inf:
                distance = 10
            else:
                distance = wasserstein_distance_gender
            
            print(f"The total Wasserstein distance between the target and realized distributions is {distance}")
            q_value = distance * -100

            ind_selected = np.argsort(similarity_matrix)[::-1][:10]
            gender_selected = []
            profession_selected = []
            for idmax in ind_selected:
                gender_selected.append(filtered_user_profiles[int(idmax)]["Gender"])
                # profession_selected.append(filtered_user_profiles[int(idmax)]["profession_name"])

            if real_female_pct > 0:
                sr_female = gender_selected.count("Female") / genders.count("Female")
            else:
                sr_female = 0
            
            if real_male_pct > 0:
                sr_male = gender_selected.count("Male") / genders.count("Male")
            else:
                sr_male = 0            

            impact_r_female = sr_female / max(sr_female, sr_male)
            impact_r_male = sr_male / max(sr_female, sr_male)
            print("IR F", impact_r_female, "IR M", impact_r_male)
            logging.info(f"Impact Ratio Female: {impact_r_female}, Impact Ratio Male: {impact_r_male}")

            for idmax in ind[:5]:
                logging.info(f"The most similar profile cosine similarity: {similarity_matrix[idmax]}")
                logging.info("=="*35)
                logging.info(filtered_user_profiles[int(idmax)]["text"])
                logging.info("=="*35)
                logging.info(f"Gender: {filtered_user_profiles[int(idmax)]['Gender']}")
                logging.info("=="*35)

            # male_selected_profession_percentages = {}
            # female_selected_profession_percentages = {}
            # male_selected_candidates = [prof for prof, gender in zip(profession_selected, gender_selected) if gender == "Male"]
            # female_selected_candidates = [prof for prof, gender in zip(profession_selected, gender_selected) if gender == "Female"]
            # total_male_selected_candidates = len(male_selected_candidates)
            # total_female_selected_candidates = len(female_selected_candidates)

            # for profession in set(male_selected_candidates):
            #     male_selected_profession_percentages[profession] = male_selected_candidates.count(profession) / total_male_selected_candidates
        
            # for profession in set(female_selected_candidates):
            #     female_selected_profession_percentages[profession] = female_selected_candidates.count(profession) / total_female_selected_candidates

        else:
            print("no match")
            wasserstein_distance_gender = 1
            distance = wasserstein_distance_gender
            q_value = -100

            real_male_pct = np.nan
            real_female_pct = np.nan

            sr_female = np.nan
            sr_male = np.nan

            impact_r_female = np.nan
            impact_r_male = np.nan

            # male_profession_percentages = {}
            # female_profession_percentages = {}
            # male_selected_profession_percentages = {}
            # female_selected_profession_percentages = {}
        logging.info(f"Q_value {q_value}")      
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
    
        return {"evaluated_text": job_desc,
                "sr_female": sr_female,
                "sr_male": sr_male,
                "gender_distance": wasserstein_distance_gender,
                "ir_female": impact_r_female,
                "ir_male": impact_r_male,
                "q_val": q_value,
                # "male_profession_percentages": male_profession_percentages,
                # "female_profession_percentages": female_profession_percentages,
                # "male_selected_profession_percentages": male_selected_profession_percentages,
                # "female_selected_profession_percentages": female_selected_profession_percentages,
                # "matched_professions": professions
                }

def main(batch_index):
    setup_logging(f"processing_batch_generated_{batch_index}.log")
    logging.info("Loading dataset...")

    logging.info("Loading evaluation dataset...")
    evaluation_dataset = load_dataset("buseskorkmaz/labelled_cleaned_hiring_w_embedding")["train"]
    print(evaluation_dataset)
    print(evaluation_dataset[0])
    batch = evaluation_dataset.select(range(batch_index * batch_size, (batch_index + 1) * batch_size))

    evaluator = Diversity_Evaluator(evaluation_dataset=batch)

    q_values = {
        "original": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='embedding')),
        "mpnet": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='mpnet')),
        "distilroberta": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='distilroberta')),
        "minilm": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='minilm')),
        "gendered_mpnet": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='gendered_embedding_mpnet')),
        "gendered_distilroberta": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='gendered_embedding_distilroberta')),
        "gendered_minilm": batch.map(lambda x: evaluator.calc_q_value(x, x["cleaned_text"], embedding_type='gendered_embedding_minilm'))
    }

    logging.info("Saving generated evaluation dataset...")

    for key, dataset in q_values.items():
        dataset.save_to_disk(f"wants_to_hired_gendered_sentence_embeddings_{key}_{batch_index}")

    logging.info(f"Sample evaluated item: {q_values['mpnet'][1]}")
    print(q_values['mpnet'][1])
    print(q_values['distilroberta'][1])
    print(q_values['minilm'][1])
    print(q_values['gendered_mpnet'][1])
    print(q_values['gendered_distilroberta'][1])
    print(q_values['gendered_minilm'][1])
    logging.info(f"Finished processing batch {batch_index}/{num_batches}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <batch_index>")
        sys.exit(1)


    batch_index = int(sys.argv[1])
    num_batches = 200
    batch_size = len(load_dataset("buseskorkmaz/labelled_cleaned_hiring_w_embedding")["train"]) // num_batches
    main(batch_index)