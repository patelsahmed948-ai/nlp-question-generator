import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Dataset class
class QGDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = self.data.loc[idx, "context"]
        answer = self.data.loc[idx, "answer"]

        input_text = f"generate question: {context} answer: {answer}"
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

# Load dataset
df = pd.read_csv("data/train/input.csv")  # Make sure input.csv exists here
train_df, val_df = train_test_split(df, test_size=0.2)

# Model & tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# DataLoader
train_dataset = QGDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "models/t5_qg.pt")
print("✅ Model saved at models/t5_qg.pt")
