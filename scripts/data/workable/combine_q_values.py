from datasets import load_from_disk, concatenate_datasets

# Load all the datasets
datasets_list = []
for i in range(200):  # Assuming you have datasets from 0 to 23
    dataset_name = f"processed_q_values_{i}"
    dataset = load_from_disk(dataset_name)
    datasets_list.append(dataset)


# Concatenate all the datasets
combined_dataset = concatenate_datasets(datasets_list)

print(combined_dataset)
print(combined_dataset[0])
# Save the combined dataset as a JSON file
combined_dataset.save_to_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions_w_q.json")
