# train_bert.py
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load GoEmotions dataset (simplified version: headline, emotion)
df = pd.read_csv("data/goemotions.csv")

# Map original emotions to our categories
mapping = {
    "fear": "fear",
    "anger": "anger",
    "annoyance": "anger",
    "excitement": "excitement",
    "joy": "excitement",
}
df["emotion"] = df["emotion"].apply(lambda x: mapping.get(x, "neutral"))

labels = df["emotion"].unique().tolist()
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

dataset = Dataset.from_pandas(df)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["headline"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("emotion", "labels")
dataset = dataset.class_encode_column("labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Train-test split
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="./models/bert_emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./models/bert_emotion")
tokenizer.save_pretrained("./models/bert_emotion")

print("✅ BERT emotion model trained and saved in models/bert_emotion/")
