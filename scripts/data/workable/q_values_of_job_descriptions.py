from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import os
import sys
random.seed(42)
# from scipy.special import kl_div
from collections import Counter
from scipy.stats import wasserstein_distance
from datasets import concatenate_datasets
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src/'))
# from hackernews.UniEval.utils import convert_to_json
# from hackernews.UniEval.metric.evaluator import get_evaluator
import nltk
nltk.download('punkt')
from transformers import BertTokenizer, BertModel
import torch
from country_code_to_location import country_code_to_main_location 

def encode_text(row):
    # Preprocess the text
    text = row['text']
    # text = text.replace('Location:', '').replace('Remote:', '').replace('Willing to relocate:', '').replace('Technologies:', '').replace('Resume:', '').replace('email:', '')
    text = text.replace('\n', ' ').replace(',', ' ')

    # Load pre-trained model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and pad the text to a maximum length of 512 tokens
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')

    # Convert to tensor
    input_ids = torch.tensor([input_ids])

    # Get the embeddings
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    # Get the embeddings of the '[CLS]' token, which represents the entire sentence
    sentence_embedding = last_hidden_states[0][0]

    # Convert the tensor to a list
    sentence_embedding = sentence_embedding.tolist()

    # Add the embedding to the row
    row['embedding'] = sentence_embedding
    return row
            
def assign_gender(row):
    gender = random.choice(['Male', 'Female'])
    row['Gender'] = gender
    return row

def extract_info(row):
    text = str(row['profile_description'])
    
    location = row['country_code']
    
    remote = re.search(r'Remote\s*:(.*?)\n', text, re.IGNORECASE)
    if remote:
        if any(term in remote.group(1).lower() for term in ['ok', 'preferred', 'yes']):
            remote = 'Yes'
        elif 'not at this time' in remote.group(1).lower():
            remote = 'No'
        else:
            remote = 'Maybe'
    else:
        remote = 'Unknown'
    
    if remote=='Unknown':
        s_location = re.search(r'Location\s*:(.*?)\n', text, re.IGNORECASE)
        if s_location and 'remote' in s_location.group(1).lower():
            remote = 'Yes'
    
    relocate = re.search(r'Willing to relocate\s*:(.*?)\n', text, re.IGNORECASE)
    if relocate:
        if any(term in relocate.group(1).lower() for term in ['ok', 'preferred', 'yes']):
            relocate = 'Yes'
        elif 'not at this time' in relocate.group(1).lower() or 'no' in relocate.group(1).lower():
            relocate = 'No'
        else:
            relocate = 'Maybe'
    else:
        relocate = 'Unknown'

    
#     us_states = [
#     'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
#     'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
#     'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
#     'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
#     'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 
#     'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 
#     'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
#     'NYC', 'San Francisco', 'Bay Area','Seattle', 'Denver', 'San Diego', 'Los Angeles', 'United States',
#     'Atlanta', 'Austin', 'Texas', 'Boston', 'Manhattan', 'America', 'Santa Barbara', 'San Jose', 'Philadelphia',
#     'Minneapolis', 'Portland', 'Miami', 'Vegas', 'Philly', 'Silicon Valley', 'Phoenix', 'Houston', 'Philidelphia', 'Dallas',
#     'Palo Alto', 'Salt Lake City', 'New Orleans', 'Salt Lake',
# ]

#     us_state_codes = [
#     'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
#     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
#     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
#     'VA', 'WA', 'WV', 'WI', 'WY', 'US', 'USA', 'Chicago', 'SF', 'DC',
# ]

#     eu_country_names = [
#     'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Republic of Cyprus', 'Czech Republic', 
#     'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Iceland',
#     'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 
#     'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'UK', 'Turkey', 'Eu', 'Europe', 
#     'England', 'Scotland', 'Wales', 'Liverpool', 'Birmingham', 'United Kingdom', 'Switzerland', 'Norway', 'Cyprus',
    
# ]

#     eu_country_capitals = [
#         'Vienna', 'Brussels', 'Sofia', 'Zagreb', 'Nicosia', 'Prague', 'Copenhagen', 'Tallinn', 
#         'Helsinki', 'Paris', 'Berlin', 'Athens', 'Budapest', 'Dublin', 'Rome', 'Riga', 'Vilnius', 
#         'Luxembourg City', 'Valletta', 'Amsterdam', 'Warsaw', 'Lisbon', 'Bucharest', 'Bratislava', 
#         'Ljubljana', 'Madrid', 'Stockholm', 'London', 'Birmingham', 'Istanbul', 'Ankara', 'Amsterdam.',
#         'Zurich', 'Oslo', 'Barcelona', 'Belgrade','Edinburgh', 'Munich', 'Milan', 'Cagliari', 'Eindhoven',
#         'Zürich' , 'Hamburg', 'Cambridge', 'Geneva', 'Genova', 'Czechia', 'Grenoble'
#     ]

#     canadian_cities = ['Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Edmonton', 'Ottawa', 'Winnipeg',
#                         'Quebec City', 'Hamilton', 'Brampton', 'Canada', 'Ontario', 'Alberta', 'Waterloo']

#     mexican_cities = ['Mexico City', 'Tijuana', 'Ecatepec', 'León', 'Puebla', 'Ciudad Juárez', 'Guadalajara',
#                        'Zapopan', 'Monterrey', 'Ciudad Nezahualcóyotl', 'Mexico', 'México']

#     south_american_countries = [
#         'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 
#         'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela', 'Costa Rica', 'Porto Rico','Guatemala', 'Jamaica',
#         'Puerto Rico'
#     ]

#     south_american_cities = ["Buenos Aires", "São Paulo", "Lima", "Bogotá", "Rio de Janeiro",
#                              "Caracas", "Santiago", "Quito", "Lima", "La Paz", "Asunción", "Montevideo", "Brasília",
#                                "Manaus", "Salvador", "Guayaquil", "Fortaleza", "Córdoba", "Recife", "Santa Cruz", "Brasil",
#                                "Panama", "Medellín"]


#     indian_cities = [ 'India', "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata","Surat",
#         "Pune", "Jaipur", "Lucknow", "Kanpur",  "Nagpur",  "Visakhapatnam", "Indore", "Thane","Bhopal","Pimpri-Chinchwad",
#         "Patna", "Vadodara"
#     ]
    
#     asian_countries = ["Afghanistan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China", 
#                     "East Timor", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan",
#                     "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
#                     "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Russia", "Saudi Arabia", 
#                     "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", 
#                     "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen", "Asia", 'Moscow', 'Minsk', 'UAE', 'Armenia',
#                     'Belarus', 'Bengaluru'
#     ]
    
#     pakistan_cities = ["Karachi", "Lahore", "Faisalabad", "Rawalpindi", "Gujranwala", "Peshawar", "Multan", "Hyderabad", "Islamabad", "Quetta"]

#     african_countries = ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", 
#                          "Central African Republic", "Chad", "Comoros", "Democratic Republic of the Congo", "Republic of the Congo", 
#                          "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", 
#                          "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", 
#                          "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", 
#                          "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
#                            "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
#     ]

#     australia_nz = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Gold Coast", 
#                     "Newcastle", "Canberra", "Wollongong", "Sunshine Coast", 'Australia', 'New Zealand', 'NZ', 'Auckland']

#     asian_capitals = ["Tokyo", "New Delhi", "Beijing", "Seoul", "Jakarta", "Islamabad", "Dhaka", "Kabul", "Nur-Sultan",
#                        "Kuala Lumpur", "Manila", "Hanoi", "Bangkok", "Taipei", "Doha", "Riyadh", "Abu Dhabi", "Muscat", "Tehran",
#                        "Hong Kong", 'Tel Aviv', 'Shanghai']
    
#     balkan_countries = ["Albania", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Greece", "Kosovo", "Montenegro",
#                          "North Macedonia", "Romania", "Serbia", "Slovenia", "Bosnia"]
#     balkan_cities = ["Tirana", "Sarajevo", "Sofia", "Zagreb", "Athens", "Prishtina", 
#                      "Podgorica", "Skopje", "Bucharest", "Belgrade", "Ljubljana"]
    
#     african_capitals = ["Algiers", "Luanda", "Porto-Novo", "Gaborone", "Ouagadougou", "Bujumbura", "Praia", "Yaoundé",
#                         "Bangui", "N'Djamena", "Moroni", "Kinshasa", "Brazzaville", "Djibouti", "Cairo", "Malabo", "Asmara", 
#                         "Addis Ababa", "Libreville", "Banjul", "Accra", "Conakry", "Bissau", "Nairobi", "Saint George's", "Bissau",
#                         "Nairobi", "Saint George's", "Malabo", "Asmara", "Addis Ababa", "Libreville", "Banjul", "Accra", "Conakry", "Bissau", 
#                         "Nairobi", "Moroni", "Maseru", "Monrovia", "Tripoli", "Antananarivo", "Lilongwe", "Bamako", "Nouakchott", "Port Louis", 
#                         "Rabat", "Maputo", "Windhoek", "Niamey", "Abuja", "Kigali", "Dakar", "Victoria", "Freetown", "Cape Town",
#                         "Mogadishu", "Pretoria", "Juba", "Khartoum", "Mbabane", "Dodoma", "Lomé", "Tunis", "Kampala", "Lusaka", "Harare", 'Lagos']



    # location = location.replace(".", "")
    # location = location.replace("/", " ")

    if location.upper() in country_code_to_main_location:
        main_location = country_code_to_main_location[location.upper()]
    else:
        main_location = 'Unknown'

    # elif any(term.lower() in location.lower() for term in us_states) or any(
    #     term.lower() in location.lower().split(",") or term.lower() in location.lower().split(" ") for term in us_state_codes
    #     ) or  any(term.lower() in location.lower() for term in canadian_cities) or any(term.lower() in location.lower() for term in mexican_cities):
    #     main_location = 'North America'
    # elif any(term.lower() in location.lower() for term in eu_country_names) or any(
    #     term.lower() in location.lower().split(",") or term.lower() in location.lower().split(" ") for term in eu_country_capitals
    #     ) or any(term.lower() in location.lower() for term in balkan_cities) or any(term.lower() in location.lower() for term in balkan_countries):
    #     main_location ='Europe'
    # # elif any(term.lower() in location.lower() for term in canadian_cities):
    # #     main_location = 'Canada'
    # # elif any(term.lower() in location.lower() for term in mexican_cities):
    # #     main_location = 'Mexico'
    # elif any(term.lower() in location.lower() for term in south_american_countries) or any(term.lower() in location.lower() for term in south_american_cities):
    #     main_location = 'South America'
    # elif any(term.lower() in location.lower() for term in indian_cities) or  any(term.lower() in location.lower() for term in asian_countries) or any(
    #     term.lower() in location.lower() for term in pakistan_cities) or any(term.lower() in location.lower() for term in asian_capitals):
    #     main_location = 'Asia'
    # elif any(term.lower() in location.lower() for term in african_countries) or any(term.lower() in location.lower() for term in african_capitals):
    #     main_location = 'Africa'
    # elif any(term.lower() in location.lower() for term in australia_nz):
    #     main_location = 'Australia'
    # elif 'Remote'.lower() in location.lower() or 'Anywhere'.lower() in location.lower() or "worldwide" in  location.lower() or "earth" in  location.lower():
    #     main_location = 'Remote'
    # else:
    #     main_location = 'Unknown'
    
    return {'Location': location, 'Remote': remote, 'Relocate': relocate, 'Main_location': main_location}

def filter_candidates(user_profile_row, job_location):
    # replace 'location' and 'remote' with the actual column names in your dataset

    # check if it is remote
    user_location = user_profile_row['Location']
    job_location = [loc for loc in job_location if loc != '' and loc != 'and']
    # print(job_location)
    if any(term in user_location.lower().split(" ") for term in job_location):
        return True
    
    # if user_profile_row['Remote'] == "Yes" or user_profile_row['Relocate'] == "Yes":
    #     return True

    # if user_profile_row['Relocate'] == "Yes":
    #     return True
    
    return False

def calc_q_value(job_desc):

    k = 50
    locations = []
    main_locations = []
    # genders = []
    # target_male_pct = 0.5
    # target_female_pct  = 0.5 
    filtered_user_profiles = user_profile_dataset.filter(lambda x: filter_candidates(x,job_desc["job_country_code"]))
    if filtered_user_profiles:
        similarity_matrix = cosine_similarity(filtered_user_profiles["embedding"], np.array(job_desc["embedding"]).reshape(1, -1)).flatten()
        # idmax = similarity_matrix.argmax()
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
        real_remote_pct = main_locations.count("Remote") / total_candidate
        real_australia_pct = main_locations.count("Australia") / total_candidate
        real_africa_pct = main_locations.count("Africa") / total_candidate
        real_sa_pct = main_locations.count("South America") / total_candidate
        real_unknown_pct =  main_locations.count("Unknown") / total_candidate

        # target_location_distribution = np.array(list(get_element_percentages(user_profile_dataset["Main_location"]).values()))
        # realized_location_distribution = np.array(list(get_element_percentages(filtered_user_profiles["Main_location"]).values()))
        target_location_distribution =  np.array([target_na_pct, target_eu_pct, target_asia_pct, target_remote_pct, target_australia_pct, target_africa_pct,
                                                target_sa_pct, target_unknown_pct])
        realized_location_distribution = np.array([real_na_pct, real_eu_pct, real_asia_pct, real_remote_pct, real_australia_pct, real_africa_pct,
                                                real_sa_pct, real_unknown_pct])

        print(target_location_distribution)
        print(realized_location_distribution)

        # Calculate the Wasserstein divergence -- always finite
        # wasserstein_distance_gender = wasserstein_distance(target_gender_distribution, realized_gender_distribution)
        wasserstein_distance_location = wasserstein_distance(target_location_distribution, realized_location_distribution)

        # print(f"The Wasserstein distance between the target gender and realized distributions is {wasserstein_distance_gender}")
        print(f"The Wasserstein distance between the target location and realized distributions is {wasserstein_distance_location}")

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

        # print("The most similar profile cosine similarity:", similarity_matrix[idmax])
        # print("=="*35)
        # print(user_profile_dataset[int(idmax)]["text"])
        # print("=="*35)
        # print("Gender:", user_profile_dataset[int(idmax)]["Gender"])
        # print("=="*35)
        # print(job_desc["text"])
    else:
        print("no match")
        q_value = -100
    
    # q_value += language_value

    print("Q_value",  q_value)
    print("--"*50, "\n\n")  

    return {"q_value": q_value }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")

    args = parser.parse_args()
    index = args.index

    print("Loading job description dataset...")
    hiring_dataset =load_dataset("json",  data_files='/dccstor/autofair/workable/job_descriptions.json')['train']
    print(hiring_dataset)

    # initialize evaluator
    # language_eval = Language_Evaluator()

    # split batches
    num_batches = 24
    batched_datasets = []
    length_of_dataset = len(hiring_dataset)

    for i in range(num_batches):
        start_index = int((i * length_of_dataset) / num_batches)
        end_index = int(((i + 1) * length_of_dataset) / num_batches)
        batched_datasets.append(hiring_dataset.shard(num_batches, i))

    # take the specified shard
    hiring_dataset = batched_datasets[index]
    hiring_dataset = hiring_dataset.map(encode_text)

    # print("Loading rl dataset...")
    # rl_dataset = load_dataset("buseskorkmaz/rl_dataset")
    # print(rl_dataset)

    print("Loading candidates dataset...")
    user_profile_dataset = load_dataset("json", data_files='/dccstor/autofair/workable/unique_candidate_database.json')['train']
    print("The downloaded version:", user_profile_dataset)
    # Map the function to the dataset
    # user_profile_dataset = user_profile_dataset.map(assign_gender)
    # print("Genders are assigned randomly:", user_profile_dataset)
    # # Map the function to the dataset
    user_profile_dataset = user_profile_dataset.map(extract_info)
    print("Extracted columns location, remote, relocate and main_location",user_profile_dataset)

    print("Calculating location diversity statistics of the dataset....")
    total_candidate = len(user_profile_dataset)
    target_na_pct = user_profile_dataset["Main_location"].count("North America") / total_candidate
    target_eu_pct = user_profile_dataset["Main_location"].count("Europe") / total_candidate
    target_asia_pct = user_profile_dataset["Main_location"].count("Asia") / total_candidate
    target_remote_pct = user_profile_dataset["Main_location"].count("Remote") / total_candidate
    target_australia_pct = user_profile_dataset["Main_location"].count("Australia") / total_candidate
    target_africa_pct = user_profile_dataset["Main_location"].count("Africa") / total_candidate
    target_sa_pct = user_profile_dataset["Main_location"].count("South America") / total_candidate
    target_unknown_pct =  user_profile_dataset["Main_location"].count("Unknown") / total_candidate

    print("Calculating q value for each job description....")
    hiring_dataset = hiring_dataset.map(calc_q_value)

    hiring_dataset.save_to_disk(f"processed_q_values_{index}")

    # print("Combining q values with rl dataset (has state and action)...")
  
    # # First, rename the column in both datasets to have a common name for joining
    # rl_dataset = rl_dataset.rename_column("your_text_column_in_rl_dataset", "text")
    # hiring_dataset = hiring_dataset.rename_column("your_text_column_in_hiring_dataset", "text")
    # print("After calculation", hiring_dataset[7])

    # # Next, perform the join operation
    # combined_dataset = rl_dataset.join(hiring_dataset, keys="text", join_type="inner")
    # print("Combined dataset", combined_dataset[7])

    # # At this point, combined_dataset contains all columns from both rl_dataset and hiring_dataset
    # # Now, let's remove unnecessary columns from combined_dataset
    # required_columns = ['state', 'action', 'q_value']
    # for column in combined_dataset.column_names:
    #     if column not in required_columns:
    #         combined_dataset = combined_dataset.remove_columns(column)

    # # Now, combined_dataset has only 'state', 'action', 'q_value', and 'text' columns
    # # And q_value for a matched text from hiring_dataset has been assigned to rl_dataset
    # rl_dataset = combined_dataset
    # print("RL dataset", rl_dataset[7])

    # print("pushing new dataset to hub")
    # rl_dataset.push_to_hub("buseskorkmaz/rl_dataset_w_q_vals", private=True)
