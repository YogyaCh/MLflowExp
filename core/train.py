import mlflow
import mlflow.sklearn
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import argparse, yaml
from sklearn.metrics import accuracy_score, f1_score
import torch

def train_model(config):
    # Load dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        logging_dir="./logs",
        logging_steps=10,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].shuffle(seed=42).select(range(1000)),
        eval_dataset=dataset["test"].shuffle(seed=42).select(range(500)),
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run():
        mlflow.log_params(config)
        trainer.train()
        trainer.save_model("model")
        mlflow.log_artifacts("model")
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_model(config)
