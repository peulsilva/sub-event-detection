
## Generating summaries
import os
while 'notebooks' in os.getcwd():
    os.chdir("..")

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, './')
import numpy as np
import pandas as pd 
from src.utils import train_test_split, get_sample_weights, get_eval_set
from src.preprocessing import preprocess_data
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.preprocessing import TextDataset
import torch
from torch.utils.data import DataLoader, Dataset
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, LoggingHandler
import logging
from copy import deepcopy
from sklearn.decomposition import PCA
from huggingface_hub import notebook_login
from sklearn.ensemble import RandomForestClassifier
from peft import get_peft_model, LoraConfig, TaskType
from collections import defaultdict
import transformers
from peft import get_peft_model, LoraConfig, TaskType
import re
from bert_score import BERTScorer
import langid
from src.utils import aggregate_samples, evaluate_model, compute_class_weights, remove_hashtag_links, get_first_texts
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BitsAndBytesConfig

tqdm.pandas()
train_data, test_data = train_test_split()

df = pd.concat(train_data)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir = '/Data')
df['tokens'] = df['Tweet'].progress_apply(tokenizer.tokenize)

target_words = [
    "goal", "penalty", "halftime", "full-time", "yellow", "red",
    "kickoff", "extra time", "stoppage time", "foul", "offside", "handball",
    "save", "tackle", "dribble", "corner", "substitution", "header",
    "free kick", "throw-in", "assist", "hat-trick", "own goal", "victory",
    "defeat", "draw", "win", "loss", "tie", "comeback", "goalkeeper",
    "striker", "midfielder", "defender", "referee", "fans", "var", "gooal"
]
target_words = set(tokenizer.tokenize(" ".join(target_words)))

def is_valid_text(t):
    for w in t:
        if w in target_words:
            return True
        
    return False

df['is_valid']= df['tokens'].progress_apply(is_valid_text)
# df['lan'] = df['Tweet'].progress_apply(lambda x : langid.classify(x)[0])
en_df = df.query("is_valid == 1")#.query("lan == 'en' ")
en_df
possible_indices = set(train_data.keys())

test_indices = list(np.random.choice(list(possible_indices), size=3, replace = False,))
test_indices = [13,1,18]
all_train_indices = list(possible_indices.difference(set(test_indices)))
val_indices = [1,5,12,19]
# val_indices = list(np.random.choice(all_train_indices, 3, replace=False))
# train_indices = list(set(all_train_indices).difference(set(val_indices)))
# train_indices = [0,2,7,11,13,18]



train_df = aggregate_samples(en_df, list(possible_indices), max_size = 10)


base_prompt = '''


You are an AI tasked with analyzing a collection of tweets posted during a single minute of a football match. Your goal is to generate a concise summary of the key events that occurred during this time and to specifically answer whether any of the following events occurred: 

1. A goal (including who scored, if mentioned).
2. A yellow or red card (including the player or team, if mentioned).
3. A kickoff (start of a half).
4. Halftime or fulltime whistle.

{tweets}

### Instructions:
1. Analyze the tweets for clear indications of the above events using common football-related terminology, phrases, or hashtags.
2. If the event is ambiguous or not explicitly stated in the tweets, mark it as "Not mentioned."
3. Summarize any additional significant match events or fan reactions from the tweets that are relevant to understanding the minute.
4. Return the response in the following structured format:

```
**Summary of Events:**
[Your summary here.]

**Specific Event Analysis:**
- Goal: [Yes/No/Not mentioned] [Additional details if yes.]
- Yellow Card: [Yes/No/Not mentioned] [Additional details if yes.]
- Red Card: [Yes/No/Not mentioned] [Additional details if yes.]
- Kickoff: [Yes/No/Not mentioned] [Additional details if yes.]
- Halftime: [Yes/No/Not mentioned]
- Fulltime: [Yes/No/Not mentioned]
```

Use this structure to organize the analysis clearly and concisely.

'''
# K Fold CV


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    cache_dir = "/Data"    
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config = quantization_config,
    device_map="auto",
    cache_dir = "/Data" 
)
# tokenizer.pad_token = tokenizer.eos_token


device = 'cuda'

generated_data = []

with torch.no_grad():
    for i, (idx, row) in enumerate(tqdm(train_df.iterrows(), total = len(train_df))):
        text = row['Tweet']
        label = row['EventType']

        match_id, period_id = idx

        prompt = base_prompt.format(tweets = text)
        message = [ {"role": "user", "content": prompt}]

        template = tokenizer.apply_chat_template(
            message,
            tokenize= False
        )

        tokens = tokenizer(
            template,
            return_tensors = 'pt'
        )

        generated_ids = base_model.generate(
            tokens['input_ids'].to('cuda'),
            # attention_mask = tokens['attention_mask'].to("cuda"),
            max_new_tokens = 200,
            do_sample = False,
            # temperature = 1.
        )

        decoded = tokenizer.batch_decode(generated_ids[:, tokens['input_ids'].shape[1]:])
        generated_t = decoded[0].split("<|end_header_id|>")[1].split("<|eot_id|>")[0]

        clear_output()
        print(f'''
            generated text : {generated_t}
            label : {label}
            '''
        )

        data = {
            'summary': generated_t,
            "label": label,
            "Tweet": text,
            "MatchID": match_id,
            "PeriodID":period_id
        }

        generated_data.append(data)

        if i % 100 == 0:
            pd.DataFrame(generated_data).to_csv("generated_predictions.csv")
pd.DataFrame(generated_data).to_csv("generated_predictions_final.csv")
# Combine results for this fold
# validation_results = pd.DataFrame({
#     "MatchID": validation_data["MatchID"].values,
#     "true_values": labels,
#     "predictions": preds,
# })

# final_results.append(validation_results)