from datasets import load_dataset
import json
import datetime

# Load the dataset
dataset = load_dataset('json', data_files='/dccstor/autofair/workable/dataset.v1.location.jsonl')
data = dataset['train']

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

# Extract candidate data
candidates = []
for entry in data:
    job_id = entry['job']['id']
    
    # Extract candidates from 'candidates' list
    for candidate in entry['candidates']:
        candidate_data = {
            'job_id': job_id,
            'applicant_id': candidate['applicant_id'],
            'country': candidate['country'],
            'country_code': candidate['country_code'],
            'experiences': candidate['experiences'],
            'educations': candidate['educations'],
            'origin': 'matched',
            'profile_description': generate_profile_description(candidate)
        }
        candidates.append(candidate_data)

    # Extract candidates from 'candidates_same_function' list
    candidates_ids = [c['applicant_id'] for c in entry['candidates']]
    for candidate in entry['candidates_same_function']:
        if candidate['applicant_id'] not in candidates_ids:
            candidate_data = {
                'job_id': job_id,
                'applicant_id': candidate['applicant_id'],
                'country': candidate['country'],
                'country_code': candidate['country_code'],
                'experiences': candidate['experiences'],
                'educations': candidate['educations'],
                'origin': 'filtered',
                'profile_description': generate_profile_description(candidate)
            }
            candidates.append(candidate_data)

    # Extract candidates from 'candidates_different_function' list
    for candidate in entry['candidates_different_function']:
        candidate_data = {
            'job_id': job_id,
            'applicant_id': candidate['applicant_id'],
            'country': candidate['country'],
            'country_code': candidate['country_code'],
            'experiences': candidate['experiences'],
            'educations': candidate['educations'],
            'origin': 'no match',
            'profile_description': generate_profile_description(candidate)
        }
        candidates.append(candidate_data)

# Custom serializer for datetime objects
def default_serializer(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

# Save the candidate dataset to a new JSON file
with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/matched_candidate_profiles.json', 'w') as outfile:
    json.dump(candidates, outfile, indent=4, default=default_serializer)


print("Candidates dataset saved to '/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/matched_candidate_profiles.json'")
