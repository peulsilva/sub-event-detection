
import torch
from torch.utils.data import DataLoader, Dataset
class TextDataset(Dataset):
    def __init__(self, texts ,labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
            # 'count' : torch.tensor(self.text_count[idx], dtype= torch.float32)
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

    return df