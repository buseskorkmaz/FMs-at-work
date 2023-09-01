from datasets import load_dataset

# Load the dataset
dataset = load_dataset('json', data_files='/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions.json')

# Display sample data
sample_index = 0  # Change this to view different samples
print("\nSample Data:")
print(f"Job: {dataset['train'][sample_index]}")
