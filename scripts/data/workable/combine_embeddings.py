from datasets import load_from_disk, concatenate_datasets

# Load all the datasets
datasets_list = []
for i in range(100):  # Assuming you have datasets from 0 to 23
    dataset_name = f"processed_profile_embeddings_{i}"
    dataset = load_from_disk(dataset_name)
    datasets_list.append(dataset)


# Concatenate all the datasets
combined_dataset = concatenate_datasets(datasets_list)

# location_distribution = {}

# for candidate in combined_dataset:
#     location = candidate['country_code']  # Assuming 'country_code' is the main location
#     if location in location_distribution:
#         location_distribution[location] += 1
#     else:
#         location_distribution[location] = 1
# print("Location_distribution", location_distribution)
# total_candidates = len(candidates_data)

print(combined_dataset)
print(combined_dataset[0])
# Save the combined dataset as a JSON file
combined_dataset.save_to_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/unique_candidates_w_embedding.json")
