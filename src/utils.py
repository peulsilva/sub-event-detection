from src.path_reader import BASE_PATH_TRAIN, BASE_PATH_TEST
from src.preprocessing import preprocess_data
from tqdm import tqdm
import pandas as pd
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from IPython.display import clear_output
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re

TRAIN_IDX = list(set([0, 2, 4, 7, 8, 11, 13, 14, 18, 19, 1, 3, 5, 10, 12, 17]))
# TRAIN_IDX = list(TRAIN_IDX.difference([4,8,10,14,17]))
# TRAIN_IDX = [4,18,19,14,13,11]
TEST_IDX = []

def train_test_split(preprocess = True):
    all_dfs = []

    for file in tqdm(os.listdir(BASE_PATH_TRAIN)):
        file_path = os.path.join(BASE_PATH_TRAIN, file)
        all_dfs.append(pd.read_csv(file_path))

    all_df = pd.concat(all_dfs)

    if preprocess:
        all_df = preprocess_data(all_df)
        

    train_df = {key: group.sort_values(by = "Timestamp") for key, group in all_df.query(f"MatchID in {TRAIN_IDX}").groupby("MatchID")}
    test_df = {key: group.sort_values(by = "Timestamp") for key, group in all_df.query(f"MatchID in {TEST_IDX}").groupby("MatchID")}

    return train_df, test_df

def get_eval_set():
    all_dfs = []

    for file in tqdm(os.listdir(BASE_PATH_TEST)):
        file_path = os.path.join(BASE_PATH_TEST, file)
        all_dfs.append(pd.read_csv(file_path))

    all_df = pd.concat(all_dfs)

    return all_df

def get_sample_weights():
    all_dfs = []

    for file in tqdm(os.listdir(BASE_PATH_TRAIN)):
        file_path = os.path.join(BASE_PATH_TRAIN, file)
        all_dfs.append(pd.read_csv(file_path))

    all_df = pd.concat(all_dfs)

    return all_df.groupby(["MatchID", "PeriodID"]).ID.count()

def get_samples_per_match():
    all_dfs = []

    for file in tqdm(os.listdir(BASE_PATH_TRAIN)):
        file_path = os.path.join(BASE_PATH_TRAIN, file)
        all_dfs.append(pd.read_csv(file_path))

    all_df = pd.concat(all_dfs)

    return all_df.groupby(["MatchID",]).ID.count()

from torch.cuda.amp import autocast
def evaluate_model(
    val_df: pd.DataFrame, 
    val_dataloader, 
    model, device : str = 'cuda', 
    use_labels = True, 
    extra_feature : bool = False,
    return_proba  =False
):
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    with torch.no_grad():
        with torch.autocast(device_type = 'cuda'):
            for i,batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader)):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                count = batch['count'].to(device).unsqueeze(dim = -1)

                labels = None
                if  use_labels:
                    labels = batch["labels"].to(device)
                # count = batch['count'].to(device).unsqueeze(dim = -1)

                if extra_feature:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, extra_feature = count)
                    preds = torch.argmax(outputs, dim=1)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                    probas = torch.softmax(outputs.logits, dim = 1)[:,1]
                    
                all_probas.extend(probas.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if use_labels:
                    all_labels.extend(labels.cpu().numpy())

                # if i % 100 == 0: 
                #     acc = accuracy_score(all_labels, all_preds)
                #     f1 = f1_score(all_labels, all_preds)

                #     clear_output()
                #     print(f"Validation Accuracy : {acc}\n")
                #     print(f"Validation F1 : {f1}\n")
                #     conf_matrix = confusion_matrix(all_labels, all_preds)
                #     print(conf_matrix)

    if use_labels:
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)

        clear_output()
        print(f"Validation Accuracy : {acc}\n")
        print(f"Validation auc : {auc}\n")
        print(conf_matrix)

    if return_proba:
        return all_preds, all_labels, probas
    return all_preds, all_labels


def compute_class_weights(labels):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def get_first_texts(x, max_size = 100, num_texts = 50):
    size = x.apply(lambda x: len(x.split(" ")))\
        .sort_values()
    
    x = x.reindex_like(size)
    mask = size < max_size
    # mask = x.str.lower().str.contains("goal")

    return "\n".join(x[mask].iloc[0:num_texts])
    # return x[mask].tolist()

def aggregate_samples(df, indices, max_size : int = 100):
    all_df = []

   
    return (df.query(f"MatchID in {indices}")).groupby(["MatchID", "PeriodID"]).agg({
        "Tweet":    lambda x: get_first_texts(x, max_size),
        "EventType": np.mean,
        "ID": len
    })

def remove_hashtag_links(df):

    df['Tweet'] = df['Tweet'].str.replace(r"#\w+", "", regex=True)

    # Remove links
    df['Tweet'] = df['Tweet'].str.replace(r"http\S+|www\S+", "", regex=True)

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric shapes extended
        u"\U0001F800-\U0001F8FF"  # Supplemental arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE
    )
    df['Tweet'] = df['Tweet'].str.replace(emoji_pattern, "", regex=True)
    df['Tweet'] = df['Tweet'].str.strip()

    # df['Tweet'] = "Is there any event like goal, halftime, fulltime, start of match or cards in any of the following tweets?\n\n" + df['Tweet']

    return df

def validate_pet_model(model, val_dataloader, tokenizer, allowed_tokens ):
    model.eval()
    val_loss = 0
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            # Forward pass

            with torch.autocast( device_type = 'cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

                # Create a mask to identify `[MASK]` positions
                mask_positions = (input_ids == tokenizer.mask_token_id)

                # Extract logits for `[MASK]` positions only
                masked_logits = logits[mask_positions]  # Shape: (num_masks, vocab_size)

                # Filter logits to include only "yes" and "no"
                allowed_logits = masked_logits[:, allowed_tokens]  # Shape: (num_masks, len(allowed_tokens))

                # Create the corresponding target labels (mapped to indices in allowed_tokens)
                target_labels = labels[mask_positions]
                remapped_labels = torch.zeros_like(target_labels)
                for i, token_id in enumerate(allowed_tokens):
                    remapped_labels[target_labels == token_id] = i

            # Compute loss

            # Store predictions and labels for evaluation
            preds = torch.argmax(allowed_logits, dim=-1)
            total_preds.extend(preds.cpu().tolist())
            total_labels.extend(remapped_labels.cpu().tolist())

    # Calculate metrics
    accuracy = accuracy_score(total_labels, total_preds)
    cnf = confusion_matrix(total_labels, total_preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(cnf)

    return total_preds, total_labels