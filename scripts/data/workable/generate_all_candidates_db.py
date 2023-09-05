from datasets import load_dataset
import json
import datetime

# Function to generate profile description
def generate_profile_description(candidate):
    experiences_desc = []
    for exp in candidate['experiences']:
        if exp['current']:
            exp_desc = f"{exp['title']} at {exp['company']}, from {exp['start_date']} to current."
        else:
            exp_desc = f"{exp['title']} at {exp['company']}, from {exp['start_date']} to {exp['end_date']}."
        experiences_desc.append(exp_desc)

    educations_desc = []
    for edu in candidate['educations']:
        edu_desc = f"Studied {edu['field_of_study']} at {edu['school_name']}, from {edu['start_date']} to {edu['end_date']}."
        educations_desc.append(edu_desc)

    profile_desc = "Experience:\n- " + "\n- ".join(experiences_desc) + "\n\n"
    profile_desc += "Education:\n- " + "\n- ".join(educations_desc)

    return profile_desc

# Load the dataset
dataset = load_dataset('json', data_files='/dccstor/autofair/workable/dataset.v1.location.jsonl')
data = dataset['train']

# Extract unique candidate data
all_candidates = []
seen_applicant_ids = set()  # To keep track of unique candidates

for entry in data:
    # Extract candidates from all lists
    for candidate_list in ['candidates', 'candidates_same_function', 'candidates_different_function']:
        for candidate in entry[candidate_list]:
            applicant_id = candidate['applicant_id']
            if applicant_id not in seen_applicant_ids:
                candidate_data = {
                    'applicant_id': applicant_id,
                    'country': candidate['country'],
                    'country_code': candidate['country_code'],
                    'experiences': candidate['experiences'],
                    'educations': candidate['educations'],
                    'profile_description': generate_profile_description(candidate),
                }
                all_candidates.append(candidate_data)
                seen_applicant_ids.add(applicant_id)  # Mark this candidate as seen

# Custom serializer for datetime objects
def default_serializer(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

# Save the candidate dataset to a new JSON file
with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/unique_candidate_database.json', 'w') as outfile:
    json.dump(all_candidates, outfile, indent=4, default=default_serializer)

print("Unique candidate database saved to 'unique_candidate_database.json'")
