
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("model1/")
tokenizer = AutoTokenizer.from_pretrained("model1/")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Input text
text = "Today is saturday, yet the management is planning classes"

# Run prediction
results = classifier(text)[0]

# Prepare data
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

labels = [emotion_labels[int(item['label'].split('_')[1])] for item in results]

scores = [item['score'] for item in results]

# Plot
plt.figure(figsize=(10, 5))
plt.barh(labels, scores)
plt.xlabel('Score')
plt.title(f"Emotion prediction for: \"{text}\"")
plt.tight_layout()
plt.show()
