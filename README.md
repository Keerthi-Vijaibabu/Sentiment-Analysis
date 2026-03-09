✅ Project Overview
Title: Sentiment Analysis using BERT and GoEmotions

Description: Fine-tuned a BERT model (bert-base-uncased) on the GoEmotions dataset to perform multi-class emotion classification. Simplified multi-label data into single-label format for compatibility with HuggingFace Transformers.

📦 Requirements 

Python 3.11
transformers
datasets
evaluate
torch
numpy
tkinter (for UI)

🧠 Model Details
Model: BERT (bert-base-uncased)

Dataset: GoEmotions

Label Handling: Multilabel simplified to single label (first label or 0)

🖥️ How to Use
🔧 Installation

git clone https://github.com/Keerthi-Vijaibabu/Sentiment-Analysis.git
cd Sentiment_Analysis
pip install -r requirements.txt

🏋️‍♀️ Training the Model

python main.py

This script will:
Load and preprocess the GoEmotions dataset
Fine-tune BERT
Save the model to model1/

📈 Predict Emotions
python pred.py
You can modify pred.py to pass your own sentences or use the provided test cases.

🖼️ GUI (Optional)
python ui.py
Launches a simple Tkinter-based UI to input text and visualize emotion predictions.

📊 Output
The model outputs emotion probabilities or labels such as:

Admiration
Amusement
Anger
Disapproval
(...full GoEmotions label list)

📌 To-Do / Future Work
Improve multilabel support
Export to ONNX / TFLite for deployment


