"""
Fine-tuning starter for Google Colab.
Input files must contain manually reviewed SQuAD-style answer spans.
This script is a baseline; validate long-context overflow handling and metrics
before claiming a completed training result.
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, DefaultDataCollator
)

MODEL_NAME = os.getenv("BASE_MODEL", "distilbert/distilbert-base-uncased")
TRAIN_FILE = os.getenv("TRAIN_FILE", "/content/qa_train.jsonl")
VALID_FILE = os.getenv("VALID_FILE", "/content/qa_validation.jsonl")
OUT_DIR = os.getenv("OUT_DIR", "/content/ruralhealth-qa")

dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

def preprocess(batch):
    enc = tokenizer(
        batch["question"], batch["context"],
        truncation="only_second", max_length=384,
        stride=96, return_offsets_mapping=True
    )
    start_positions, end_positions = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        answer = batch["answers"][i]
        if not answer["text"]:
            start_positions.append(0)
            end_positions.append(0)
            continue
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        seq_ids = enc.sequence_ids(i)
        context_ids = [j for j, sid in enumerate(seq_ids) if sid == 1]
        if not context_ids:
            start_positions.append(0); end_positions.append(0); continue
        first, last = context_ids[0], context_ids[-1]
        if offsets[first][0] > start_char or offsets[last][1] < end_char:
            start_positions.append(0); end_positions.append(0); continue
        start_tok = first
        while start_tok <= last and offsets[start_tok][0] <= start_char:
            start_tok += 1
        start_tok -= 1
        end_tok = last
        while end_tok >= first and offsets[end_tok][1] >= end_char:
            end_tok -= 1
        end_tok += 1
        start_positions.append(start_tok)
        end_positions.append(end_tok)
    enc.pop("offset_mapping")
    enc["start_positions"] = start_positions
    enc["end_positions"] = end_positions
    return enc

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    report_to="none"
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer
)
trainer.train()
trainer.save_model(OUT_DIR + "/final")
tokenizer.save_pretrained(OUT_DIR + "/final")
print("Saved model. Run held-out evaluation before reporting performance.")
