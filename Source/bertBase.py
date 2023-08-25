import numpy as np
import pandas as pd
import transformers
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch 
from pathlib import Path 
filepath = Path('../output/bertBaseOut.csv')  

MAX_LEN = 150
RANDOM_SEED = 2

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') 
 
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


df = pd.read_csv('../input/tiktok_google_play_reviews.csv')
df = df.dropna(subset='content',axis=0) 
print(df.shape)

scorecount = [0,0,0,0,0]
def count_rating(rating):
    rating = int(rating)
    scorecount[rating-1] = scorecount[rating-1] + 1
df.score.apply(count_rating)

cap = max(scorecount)                                               #visualization of rating counts from the entire dataset
print("rating \t|     count\t|     bar")
for i in range(5):
    print("\t|\t\t|")
    num = 100 * scorecount[i]/cap
    num = round(num) 
    out = "  -"
    for j in range(num):
        out = out + "-"
    print("  "+str(i+1)+"\t|     "+str(scorecount[i]) + "\t|"+ out)

def to_sentiment(rating):
    rating = int(rating)
    
    if rating <= 2:
        return -1
    elif rating == 3:
        return 0
    else:
        return 1

# Apply to the dataset 
df['sentiment'] = df.score.apply(to_sentiment)


def compute_sentiment(review):
    tokens = tokenizer.encode(review, return_tensors='pt') 
    result = model(tokens) 
    temp = 0 
    temp = int(torch.argmax(result.logits))+1 
    if temp == 1 or temp == 2: 
        return -1 
    elif temp == 4 or temp == 5: 
        return 1 
    else: 
        return 0 

def to_class(sentiment):
    if sentiment == -1:
        return 'negative'
    elif sentiment == 0:
        return 'neutral'
    else:
        return 'positive'

df = df[:12495]
class_names = ['Negative', 'Neutral', 'Positive']
df['computed_sentiment'] = df.content.apply(lambda x: compute_sentiment(x[:MAX_LEN]))
df['computed_sentiment'] = df.computed_sentiment.apply(to_class)
df['sentiment'] = df.sentiment.apply(to_class)
print(df[["content","sentiment","computed_sentiment"]].head)
print(classification_report(df["sentiment"], df["computed_sentiment"], target_names=class_names))

df["pred_class"] = df["computed_sentiment"]
df["class"] = df["sentiment"]
filepath.parent.mkdir(parents=True, exist_ok=True)  #save predictions to a file
df[["content","class","pred_class"]].to_csv(filepath)  
