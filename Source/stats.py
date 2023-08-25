import numpy as np
import pandas as pd
from pathlib import Path 
filepath = Path('../input/reviews_evened.csv')  
df = pd.read_csv('../input/tiktok_google_play_reviews.csv')

def to_sentiment(rating):
    rating = int(rating)
    
    # Convert to class
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2
    
def to_posneg(sentiment):
    sentiment = int(sentiment)
    
    # Convert to class
    if sentiment ==0:
        return "negative"
    elif sentiment == 1:
        return "neutral"
    else:
        return "positive"

# Apply to the dataset 
df['sentiment'] = df.score.apply(to_sentiment)

df['class'] = df.sentiment.apply(to_posneg)

scorecount = [0,0,0,0,0]
def count_rating(rating):
    rating = int(rating)
    scorecount[rating-1] = scorecount[rating-1] + 1
df.score.apply(count_rating)
print(df[["content","class","sentiment"]].sample(n=20))
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
    
total = sum(scorecount)
negative = scorecount[0] + scorecount[1]
neutral = scorecount[2]
positive = scorecount[3] + scorecount [4]
print("\nOriginal data:")
print(f'counts of positive = {positive}, neutral = {neutral}, negative = {negative}')
print(f'distribution is {100*positive/total}% positive {100*neutral/total}% neutral {100*negative/total}% negative')
print(f'negative/positive = {negative/positive}')

df = df.drop(df[df['sentiment'] == 2].sample(frac=0.8207).index)      #remove some random positive reviews to have an even distribution for positive and negative 
scorecount = [0,0,0,0,0]
df.score.apply(count_rating)
total = sum(scorecount)
negative = scorecount[0] + scorecount[1]
neutral = scorecount[2]
positive = scorecount[3] + scorecount [4]
print("\nNow with the evened out data:")
print(f'counts of positive = {positive}, neutral = {neutral}, negative = {negative}')
print(f'distribution is {100*positive/total}% positive {100*neutral/total}% neutral {100*negative/total}% negative')
print(f'negative/positive = {negative/positive}')
# filepath.parent.mkdir(parents=True, exist_ok=True)  #save evened distribution to a file
# df.to_csv(filepath)  