import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, './')

from src.utils import train_test_split
from src.preprocessing import TextDataset
from tqdm import tqdm
import pandas as pd
from IPython.display import clear_output

# Initialize tokenizer and model cache path
cache_dir = '/Data'
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


# Sample Data Splitting Function

def compute_class_weights(labels):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

# Training Function
def train_model(train_dataloader, model, optimizer, loss_fn,n_epochs=10, device='cuda', accumulation_steps=1):
    model.train()
    for epoch in range(n_epochs):
        all_preds = []
        all_labels = []
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = loss_fn(logits, labels) / accumulation_steps
                # loss = outputs.loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            epoch_loss += loss.item()

            if i % 10 == 0:
                clear_output()
                print(f"Epoch {epoch}")
                print(f"Batch accuracy: {accuracy_score(all_labels, all_preds)}")
                print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

                all_labels = []
                all_preds = []

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        clear_output()
        print(f"---------- Epoch {epoch+1} ------------")
        print(f"Training Loss: {epoch_loss}")
        print(f"Training Accuracy: {acc}")
        print(f"Training F1 Score: {f1}")

# Evaluation Function
def evaluate_model(val_dataloader, model, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_preds, all_labels)
    f1 = f1_score(all_preds, all_labels)
    print(f"Validation Accuracy: {acc}")
    print(f"Validation F1 Score: {f1}")
    return all_preds, all_labels

# Main K-Fold CV Process
def main_kfold_cv(device='cuda'):
    final_results = []
    train_dict, test_dict = train_test_split()
    train_df = pd.concat(train_dict.values())

    for validation_set_id, validation_data in train_dict.items():
        train_data = train_df.query(f"MatchID != {validation_set_id}")

        # Prepare datasets and dataloaders
        print("Tokenizing train")
        train_dataset = TextDataset(
            train_data["Tweet"].tolist(), train_data["EventType"].tolist(), tokenizer
        )
        print("Tokenizing train")
        val_dataset = TextDataset(
            validation_data["Tweet"].tolist(), validation_data["EventType"].tolist(), tokenizer
        )

        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=512)

        labels = train_data["EventType"].tolist()
        class_weights = compute_class_weights(labels).to(device)

        # Define weighted loss function
        loss_fn = torch.nn.CrossEntropyLoss()


        # Load and configure model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir=cache_dir
        ).to(device)

        # for param in model.base_model.parameters():
        #     param.requires_grad = False


        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence classification task
            r=16,  # LoRA rank
            lora_alpha=16,  # Scaling factor
            lora_dropout=0.1  # Dropout for LoRA layers
        )
        model = get_peft_model(base_model, lora_config).to(device)


        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # Train the model
        train_model(train_dataloader, model, optimizer, loss_fn ,n_epochs=5)

        # Evaluate the model
        preds, labels = evaluate_model(val_dataloader, model)

        # Collect results for the fold
        validation_results = pd.DataFrame({
            "MatchID": validation_data["MatchID"].values,
            "true_values": labels,
            "predictions": preds,
        })
        final_results.append(validation_results)

    # Save final results to CSV
    final_results_df = pd.concat(final_results, axis=0, ignore_index=True)
    final_results_df.to_csv("transformers_final_results.csv", index=False)

main_kfold_cv()