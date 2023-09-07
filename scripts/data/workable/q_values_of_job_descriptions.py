import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import sys
random.seed(42)
from scipy.stats import wasserstein_distance
from datasets import concatenate_datasets, load_from_disk, load_dataset, set_caching_enabled
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src/'))
import nltk
nltk.download('punkt')
import torch
import pycountry
import pandas as pd
import geopandas as gpd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

set_caching_enabled(False)

def alpha2_to_name(alpha_2):
    try:
        return pycountry.countries.get(alpha_2=alpha_2).name
    except LookupError:
        return None
    except AttributeError:
        return None
            
# def assign_gender(row):
#     gender = random.choice(['Male', 'Female'])
#     row['Gender'] = gender
#     return row

def extract_info(row):
    location_name = row['country']

    if location_name == "United States":
        location_name = "United States of America"

    df = pd.DataFrame(data=[location_name], columns=["country_name"])
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    merged = world.merge(df, left_on='name', right_on='country_name', how='right')
    # print(merged)

    if merged['continent'].values[0] != np.nan:
        main_location = merged['continent'].values[0]
        # print("Location", location_name, "Main location:", main_location)
    else:
        main_location = 'Unknown'
    
    row['Location'] = str(location_name)
    row['Main_location'] = str(main_location)

    return row

def filter_candidates(user_profile_row, job_country_code):
    # both are country names
    user_location = user_profile_row['Location'].lower()
    if user_location == "United States":
        user_location = "United States of America"

    job_location = alpha2_to_name(job_country_code).lower()
    
    return user_location == job_location

def calc_q_value(job_desc):

    k = 50
    locations = []
    main_locations = []
    # genders = []
    # target_male_pct = 0.5
    # target_female_pct  = 0.5 
    # filtered_user_profiles = user_profile_dataset.filter(lambda x: filter_candidates(x,job_desc["job_country_code"]))
    filtered_user_profiles = user_profile_dataset
    if filtered_user_profiles:
        similarity_matrix = cosine_similarity(filtered_user_profiles["embedding"], np.array(job_desc["embedding"]).reshape(1, -1)).flatten()
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

        target_location_distribution =  np.array([target_na_pct, target_eu_pct, target_asia_pct, target_oceania_pct, target_africa_pct,
                                                target_sa_pct, target_antarctica_pct])
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

    print("Q_value",  q_value)
    # print("--"*50, "\n\n")  

    return {"q_value": q_value }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")

    args = parser.parse_args()
    index = args.index

    print("Loading job description dataset...")
    hiring_dataset =load_from_disk('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_description_w_embedding.json')
    print(hiring_dataset)

    # split batches
    num_batches = 200
    batched_datasets = []
    length_of_dataset = len(hiring_dataset)

    for i in range(num_batches):
        start_index = int((i * length_of_dataset) / num_batches)
        end_index = int(((i + 1) * length_of_dataset) / num_batches)
        batched_datasets.append(hiring_dataset.shard(num_batches, i))

    # take the specified shard
    hiring_dataset = batched_datasets[index]

    # - embeddings are ready
    # hiring_dataset = hiring_dataset.map(encode_text)

    print("Loading candidates dataset...")
    user_profile_dataset = load_from_disk('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/unique_candidates_w_embedding.json')
    print("The loaded version:", user_profile_dataset)
    # Map the function to the dataset
    # user_profile_dataset = user_profile_dataset.map(assign_gender)
    # print("Genders are assigned randomly:", user_profile_dataset)
    # # Map the function to the dataset
    user_profile_dataset = user_profile_dataset.map(extract_info, num_proc=4)
    print("Extracted columns location, and main_location",user_profile_dataset)
    user_profile_dataset.save_to_disk("candidates_w_main_locataion")

    print("Calculating location diversity statistics of the dataset....")
    total_candidate = len(user_profile_dataset)
    target_na_pct = user_profile_dataset["Main_location"].count("North America") / total_candidate
    target_eu_pct = user_profile_dataset["Main_location"].count("Europe") / total_candidate
    target_asia_pct = user_profile_dataset["Main_location"].count("Asia") / total_candidate
    target_oceania_pct = user_profile_dataset["Main_location"].count("Oceania") / total_candidate
    target_africa_pct = user_profile_dataset["Main_location"].count("Africa") / total_candidate
    target_sa_pct = user_profile_dataset["Main_location"].count("South America") / total_candidate
    target_antarctica_pct =  user_profile_dataset["Main_location"].count("Antarctica") / total_candidate

    print("Calculating q value for each job description....")
    hiring_dataset = hiring_dataset.map(calc_q_value)

    hiring_dataset.save_to_disk(f"processed_q_values_{index}")