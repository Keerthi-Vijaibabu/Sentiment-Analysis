import tkinter as tk
from tkinter import ttk
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("model1")
tokenizer = AutoTokenizer.from_pretrained("model1")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Emotion labels (GoEmotions)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# GUI setup
root = tk.Tk()
root.title("Emotion Classifier")
root.geometry("1000x800")

tk.Label(root, text="Enter a sentence:", font=("Helvetica", 12)).pack(pady=10)
entry = tk.Entry(root, width=100, font=("Helvetica", 12))
entry.pack(pady=5)

# Top-3 result label
top3_label = tk.Label(root, text="", font=("Helvetica", 12), justify=tk.LEFT)
top3_label.pack(pady=10)

# Frame to hold charts
chart_frame = tk.Frame(root)
chart_frame.pack(fill=tk.BOTH, expand=True)

def analyze():
    sentence = entry.get()
    if not sentence.strip():
        top3_label.config(text="Please enter a sentence.")
        return

    results = classifier(sentence)[0]
    scores = [r["score"] for r in results]
    labels = [emotion_labels[int(r["label"].split("_")[1])] for r in results]

    # Sort by score
    combined = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)

    # Top 3 for display
    top_3 = combined[:3]
    top3_text = "\n".join(f"{label}: {score:.2f}" for label, score in top_3)
    top3_label.config(text=f"Top 3 Emotions:\n{top3_text}")

    # Top 5 for pie chart
    top_5 = combined[:5]
    other_score = sum(score for _, score in combined[5:])
    pie_labels = [label for label, _ in top_5] + (["Others"] if other_score > 0 else [])
    pie_scores = [score for _, score in top_5] + ([other_score] if other_score > 0 else [])

    # Clear previous charts
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # Create side-by-side charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart with legend (no labels on wedges)
    wedges, texts, autotexts = ax1.pie(
        pie_scores,
        startangle=140,
        autopct='%1.1f%%',
        textprops=dict(color="white")
    )
    ax1.legend(wedges, pie_labels, title="Emotions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax1.set_title("Top 5 Emotions (Pie Chart)")
    ax1.axis("equal")

    # Bar chart: All emotions
    ax2.barh(labels, scores)
    ax2.set_xlabel("Score")
    ax2.set_title("Emotion Scores (Bar Chart)")
    ax2.invert_yaxis()

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Button
tk.Button(root, text="Analyze Emotion", font=("Helvetica", 12), command=analyze).pack(pady=10)

root.mainloop()
