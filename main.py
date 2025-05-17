from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
print(torch.cuda.is_available())  # should return True


dataset = load_dataset("go_emotions")
print(dataset['train'][0])

#converting the multilabel tokens to single label as the transformer couldn't be trained
#the tensors should be of equal size

def convert_to_single_label(example):
    example["labels"] = example["labels"][0] if example["labels"] else 0  # default to 0 if empty
    return example

dataset = dataset.map(convert_to_single_label)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=28  # 28 emotions in GoEmotions
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy = 'epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("model1/")
tokenizer.save_pretrained("model1/")  # save tokenizer too
