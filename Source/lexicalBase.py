import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path 
filepath = Path('../output/lexicalBaseOut.csv')  

import nltk
nltk.download('vader_lexicon',download_dir="../input")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


df = pd.read_csv('../input/reviews_evened.csv')
# df = pd.read_csv('../input/tiktok_google_play_reviews.csv')
# df = df.sample(n=1245) #cut dataset down for testing

df = df.dropna(subset='content',axis=0) 
class_names = ['Negative', 'Neutral', 'Positive']
# example = df[["content"]].sample(n=1)

# Using polarity scores for knowing the polarity of each text
def SentimentIntensityAnalyzer(sentence):
    score = analyser.polarity_scores(sentence)
    # print("{:-<150} {}".format(sentence, str(score)))
    
# print(example.values[0])
# print (SentimentIntensityAnalyzer(example.values[0][0]))
# print(example.values[0][1])
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df['content'].apply(tokenizer.tokenize)

all_words = [word for tokens in words_descriptions for word in tokens]
df['description_lengths']= [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))

from collections import Counter
count_all_words = Counter(all_words)

df['scores'] = df['content'].apply(lambda review: analyser.polarity_scores(review))

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

def Sentimnt(x):
    if x>= 0.01:
        return "positive"
    elif x<= -0.01:
        return "negative"
    else:
        return "neutral"
df['pred_class'] = df['compound'].apply(Sentimnt)

var1 = df.groupby('pred_class').count()['content'].reset_index().sort_values(by='content',ascending=False)
# print(var1.head())
# print(df[["content","true_class","pred_class"]].head())

print(classification_report(df["class"], df["pred_class"], target_names=class_names))
filepath.parent.mkdir(parents=True, exist_ok=True)  #save evened distribution to a file
df[["content","class","pred_class"]].to_csv(filepath)  
