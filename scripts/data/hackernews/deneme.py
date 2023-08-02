from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# hiring_dataset = load_dataset("buseskorkmaz/wants_to_be_hired_gendered")
# print(hiring_dataset)

# from sklearn.metrics.pairwise import cosine_similarity

# def encode_text(job_desc):

#     # Load pre-trained model and tokenizer
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


#     text = job_desc
#     # Preprocess the text
#     text = text.replace('\n', ' ').replace(',', ' ')

#     # Tokenize and pad the text to a maximum length of 512 tokens
#     input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')

#     # Convert to tensor
#     input_ids = torch.tensor([input_ids])

#     # Get the embeddings
#     with torch.no_grad():
#         last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

#     # Get the embeddings of the '[CLS]' token, which represents the entire sentence
#     sentence_embedding = last_hidden_states[0][0]

#     # Convert the tensor to a list
#     sentence_embedding = sentence_embedding.tolist()

#     return sentence_embedding   

# here = np.array(encode_text("""Berkeley/San Francisco - Full-time
# Captricity transforms stacks of paper forms into structured data faster and cheaper than anyone else. Our technology can greatly improve the efficiency of low-resource organizations working in the world's most disadvantaged communities. Our first customers work in Mali, Ghana and Sierra Leone. We use computer vision, machine learning and information-theoretic principles to guide an army of online workers (MTurk for now). Our workers love our tasks. Captricity has 5 full-timers, top-notch investors and seed-funding. The product is in private-Beta.
# If you want to save the world, and have shipped complex, production web-services / user-experiences, drop us a line -- we're looking for:
# 1) a senior engineer to be the lead member of our technical staff.
# 2) our first full-time UX designer

# More on the role of data in low-resource communities:"""))
# there = np.array(encode_text("""Berkeley/San Francisco
# Cricity transforms stacks of paper forms into structured. Our technology can greatly improve the efficiency of low-resource organizations working in the world's most challenging communities. Our first customers work in the USA, Canada and New York. We use computer vision, machine learning and information-theoric principles to guide an army of online workers (MTurk for now). Our workers love our tasks. to the full-timers, top-notch investors and seed-funding. The product is in private-Alpha.
# If you want to save the world, and have shipped complex.
# 1) a senior engineer to be the lead member of our technical staff.
# 2) our first full-time UX designer

# More on the role of the detail in low-'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""))
# print(cosine_similarity(here.reshape(1, -1), there.reshape(1, -1)))
