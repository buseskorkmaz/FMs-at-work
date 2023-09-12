from datasets import load_from_disk
from langdetect import detect

def lang_detect(text):
    try:
        return {"language": detect(text)}
    except:
        return {"language": None}

def en_filter(language):

    return language == "en"

hiring_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/job_descriptions_w_q_prompt")
print(hiring_dataset)

hiring_dataset = hiring_dataset.map(lambda x: lang_detect(x["description"]))
hiring_dataset = hiring_dataset.filter(lambda x: en_filter(x["language"]))
hiring_dataset = hiring_dataset.remove_columns('language')

print(hiring_dataset)

hiring_dataset.save_to_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/job_descriptions_w_q_prompt_eng")