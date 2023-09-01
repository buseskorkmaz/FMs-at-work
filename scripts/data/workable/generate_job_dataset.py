from datasets import load_dataset
import json
from bs4 import BeautifulSoup

def html_to_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

# Load the dataset
dataset = load_dataset('json', data_files='/dccstor/autofair/workable/dataset.v1.location.jsonl')
data = dataset['train']

# Extract job data
jobs = []
for entry in data:
    try:
        description = html_to_text(entry['job']['description'])   
        job = {
            'id': entry['job']['id'],
            'account_id': entry['job']['account_id'],
            'title': entry['job']['title'],
            'description': description,
            'requirement_summary': entry['job']['requirement_summary'],
            'benefit_summary': entry['job']['benefit_summary'],
            'employment_type': entry['job']['employment_type'],
            'industry': entry['job']['industry'],
            'experience': entry['job']['experience'],
            'function': entry['job']['function'],
            'education': entry['job']['education'],
            'remote': entry['job']['remote'],
            'account_name': entry['job']['account_name'],
            'job_country_code': entry['job']['job_country_code']
        }
        jobs.append(job)    
    except:
        print("printing the error")
        print(entry['job']['description'])


# Save the job dataset to a new JSON file
with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions.json', 'w') as outfile:
    json.dump(jobs, outfile, indent=4)

print("Job dataset saved to '/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions.json'")

