from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys

def predict(text):
    model = AutoModelForSequenceClassification.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("model")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

if __name__ == "__main__":
    text = sys.argv[1]
    prediction = predict(text)
    print(f"Prediction: {prediction} ({'positive' if prediction else 'negative'})")
