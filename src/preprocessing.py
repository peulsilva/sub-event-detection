
import torch
from torch.utils.data import DataLoader, Dataset
class TextDataset(Dataset):
    def __init__(self, texts, count ,labels, tokenizer, match_id, max_length=4096):
        self.texts = texts
        self.labels = labels
        self.count = count
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.match_id = match_id
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is None:
            label = -1
        else:
            label = self.labels[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # encoding = self.tokenizer(
        #     text, return_tensors="pt"
        # )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
            'count' : torch.tensor(self.count[idx], dtype= torch.float32),
            "match_id" : self.match_id[idx]
        }
    
def preprocess_data(df):
    # removes duplicates and RTs
    df = df.drop_duplicates("Tweet")
    rt_mask = df['Tweet']\
        .str\
        .lower()\
        .str\
        .contains("rt")
    df = df[~rt_mask]

    at_mask = df['Tweet']\
        .str\
        .lower()\
        .str\
        .contains("@")
    
    df = df[~at_mask]

    http_mask = df['Tweet']\
        .str\
        .lower()\
        .str\
        .contains("http")
    
    df = df[~http_mask]

    return df