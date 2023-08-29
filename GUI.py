import re
from bs4 import BeautifulSoup
import string
import re
import contractions
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tkinter as tk

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained('Model')


def remove_html_tags(text): 
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    return clean_text
    
def remove_punctuation(text):
    clean_text = "".join([char for char in text if char not in string.punctuation])
    return clean_text

def remove_special_characters(text):
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return clean_text
    
def normalize_text(text):
    clean_text = text.lower()
    return clean_text

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def remove_numbers(text):
    clean_text = re.sub(r"\d+", "", text)
    return clean_text


def remove_extra_whitespace(text):
    clean_text = " ".join(text.split())
    return clean_text



def preprocess_text(text):

    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = normalize_text(text)
    text = expand_contractions(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)

    return text

# Dark mode color scheme
BG_COLOR = '#333333'

TEXT_COLOR = "white"

BUTTON_COLOR = '#555555'

def classify_text():
    user_input = text_input.get("1.0", "end-1c")
    
    # Preprocess text
    user_input = preprocess_text(user_input)

    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    sentiment = "Positive" if predicted_class == 1 else "Negative"

    if sentiment == "Positive" :
        result_label.config(text = 'Sentiment: Positive', fg="green")
    else: 
        result_label.config(text = 'Sentiment: Negative', fg="red")


root = tk.Tk()
root.title("Text Sentiment Classifier")
root.configure(bg=BG_COLOR)

root.iconbitmap("icon.ico")

# Window size
window_width = 400
window_height = 300
root.geometry(f"{window_width}x{window_height}")

# Create a custom font size and style
custom_font = ("Arial", 12, "bold")
title_font = ('Arial', 16, 'bold') # for the title

title_label = tk.Label(root, text = 'Enter your text', font = title_font, bg=BG_COLOR, fg=TEXT_COLOR)
title_label.pack(fill= tk.X)

text_input = tk.Text(root, height=10, width=40, bg=BG_COLOR, fg=TEXT_COLOR , font = custom_font, wrap=tk.WORD)
text_input.pack()

submit_button = tk.Button(root, text="Submit", command=classify_text, bg=BUTTON_COLOR, fg=TEXT_COLOR, font = custom_font)
submit_button.pack()

 
result_label = tk.Label(root, text="", bg=BG_COLOR, fg=TEXT_COLOR,font = custom_font)
result_label.pack()

root.mainloop()
