import json


with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/unique_candidate_database.json', 'r') as infile:
    candidates_data = json.load(infile)

location_distribution = {}

for candidate in candidates_data:
    location = candidate['country_code']  # Assuming 'country_code' is the main location
    if location in location_distribution:
        location_distribution[location] += 1
    else:
        location_distribution[location] = 1

# total_candidates = len(candidates_data)
# for location, count in location_distribution.items():
#     location_distribution[location] = (count / total_candidates) * 100

with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/candidate_location_distribution.json', 'w') as outfile:
    json.dump(location_distribution, outfile, indent=4)

print("Location distributions saved to 'location_distribution.json'")


with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions.json', 'r') as infile:
    job_data = json.load(infile)

location_distribution = {}

for job in job_data:
    location = job['job_country_code']  # Assuming 'country_code' is the main location
    if location in location_distribution:
        location_distribution[location] += 1
    else:
        location_distribution[location] = 1

# total_candidates = len(job_data)
# for location, count in location_distribution.items():
#     location_distribution[location] = (count / total_candidates) * 100

with open('/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_location_distribution.json', 'w') as outfile:
    json.dump(location_distribution, outfile, indent=4)

print("Location distributions saved to 'location_distribution.json'")