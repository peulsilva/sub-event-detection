import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, \
  BitsAndBytesConfig, GPTQConfig
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from transformers import BitsAndBytesConfig
import torch
from IPython.display import clear_output
import json
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, './')
from src.utils import train_test_split

# Ensure current directory is set correctly
while 'notebooks' in os.getcwd():
    os.chdir("..")

# Configuration for quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Tokenizer and model setup
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    cache_dir="/Data"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="eager",
    cache_dir="/Data"
)



# Define train and validation data
train_indices = [8,11,13]
val_indices = [8, 11, 13]

# Assuming `train_test_split` splits data properly
train_data, test_data = train_test_split()

# Function to get samples from data
def get_samples(indices, frac=1):
    all_df = []
    for id in indices:
        temp_df = train_data[id]
        all_df.append(temp_df.dropna().sample(frac=frac))
    return pd.concat(all_df)

train_df = get_samples(train_indices)
val_df = get_samples(val_indices)

# Define the base prompt for generation
base_prompt = '''You are a helpful assistant.
    I will provide you a tweet that happened during a world cup game and you will return me if this tweet represents a football event such as a goal, half time, kick-off, full time, penalty, red card, yellow card, or
own goal—occurred within that period.

    Here is the text:
    
    {text}

    Answer in the following format (JSON):

    {{
        'event': yes or no (depending on the event),
        'reason' : a short reason (no more than 50 words)
    }}.

    Please do not output anything else than the JSON
'''

generated_samples = []
save_every = 100

# Main loop to generate responses and process JSON
for i, (idx, row) in tqdm(enumerate(train_df.sample(frac=1, replace=False).iterrows()), total = len(train_df)):
    text = row['Tweet']
    prompt = base_prompt.format(text=text)
    message = [{"role": "user", "content": prompt}]

    template = tokenizer.apply_chat_template(
        message,
        tokenize=False
    )

    tokens = tokenizer(template, return_tensors='pt')

    generated_ids = model.generate(
        tokens['input_ids'].to('cuda'),
        max_new_tokens=100,
        do_sample=False
    )

    decoded = tokenizer.batch_decode(generated_ids[:, tokens['input_ids'].shape[1]:])
    generated_t = decoded[0].split("<|end_header_id|>")[1].split("<|eot_id|>")[0]

    try:
        json_obj = json.loads(generated_t)
        json_obj['MatchID'] = row['MatchID']
        json_obj['PeriodID'] = row['PeriodID']
        json_obj['originalTweet'] = text
        json_obj['EventType'] = row['EventType']
        generated_samples.append(json_obj)
    except Exception as e:
        print("Error in JSON generation:", e)

    if 'yes' in decoded[0]:
        clear_output()
        print(f'''
            Generated text: {decoded[0]}
            Original text: {text}
            Progress: {i} out of {len(train_df)}
        ''')

    if i % save_every == 0:
        pd.DataFrame(generated_samples).to_csv("data/real_labels.csv")

pd.DataFrame(generated_samples).to_csv("data/real_labels_final.csv")
