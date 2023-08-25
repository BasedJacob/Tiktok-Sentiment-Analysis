import numpy as np
import pandas as pd
import zipfile
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path 
filepath = Path('../output/lexicalBootstrapOut.csv')  

import nltk
nltk.download('vader_lexicon',download_dir="../input")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
RANDOM_SEED=42
analyser = SentimentIntensityAnalyzer()

lex_dict = analyser.make_lex_dict()
# print(len(lex_dict),"words in dictionary")
df = pd.read_csv('../input/reviews_evened.csv')

df = df.dropna(subset='content',axis=0) 
class_names = ['Negative', 'Neutral', 'Positive']

df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df_train['content'].apply(tokenizer.tokenize)

all_words = [word for tokens in words_descriptions for word in tokens]
df_train['description_lengths']= [len(tokens) for tokens in words_descriptions]

VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))

from collections import Counter
count_all_words = Counter(all_words)

# Using polarity scores for knowing the polarity of each text
def sentiment_analyzer_score(sentence):
    score = analyser.polarity_scores(sentence)
    # print("{:-<150} {}".format(sentence, str(score)))
sentimentThresh = 0.05
def Sentimnt(x):
    if x >= sentimentThresh:
        return "positive"
    elif x <= -sentimentThresh:
        return "negative"
    else:
        return "neutral"


def defaultval():
    return [0,0,0,0,0]
def bootstrap():
    itercap = 5
    falloff = 0.007 #tentative parameters
    subjectiveThresh = 0.2
    minCount = 50
    numAdditionsPerIter = 150

    iter = 0
    stop = False

    while not stop:
        additions_to_lex = {}
        unknownwords = set(all_words) - set(lex_dict)
        unkWordDict = defaultdict(defaultval)# [key] = [subjectivity prob, positivity prob, subjectivity count, positivity count, count of appearances]
        unkWordDict.clear()
        sdict = {}
        sdict.clear()
        print("\niteration",iter+1)
        # print("num of unknown words:",len(unknownwords))
        for unk in unknownwords:
            if count_all_words[unk] > minCount:
                for sentence in words_descriptions:
                    if unk in sentence:
                        for word in sentence:
                            subjective = False
                            # positive = False
                            if word in lex_dict:
                                # if lex_dict[word] > 0:
                                #     positive = True
                                subjective = True
                        if subjective == True:
                            if (analyser.polarity_scores(''.join(sentence)))['compound']>sentimentThresh: #if the analyser says the sentence is positive
                                unkWordDict[unk][3] = unkWordDict[unk][3] + 1   #mark the positivity count
                            unkWordDict[unk][2] = unkWordDict[unk][2] + 1
                            
                        unkWordDict[unk][4] = unkWordDict[unk][4] + 1   #mark the count of appearances
                unkWordDict[unk][0] = unkWordDict[unk][2] / unkWordDict[unk][4] - iter*falloff #subjectivity prob
                if(unkWordDict[unk][2] == 0):
                    unkWordDict[unk][1]=0
                else:
                    unkWordDict[unk][1] = unkWordDict[unk][3] / unkWordDict[unk][2] #positivity prob
        sdict = dict(sorted(unkWordDict.items(), key=lambda item: item[1][0]))   #sort by subjectivity
        k = 0
        
        for item in reversed(sdict.items()):    #iterate through decreasing subjectivity
            # print(item)
            if item[1][0] > subjectiveThresh:   #check against the minimum subjectivity allowed
                if item[1][1] > 0.5:
                    additions_to_lex[item[0]] = 1   #set the polarity
                else:
                    additions_to_lex[item[0]] = -1
            k = k+1
            if k >= numAdditionsPerIter:
                # print("last subjectivity:",item[1][0])
                break
                                    
        iter = iter +1
        print("num additions to lexicon:", len(additions_to_lex))
        if len(additions_to_lex) == 0 or iter>=itercap:
            stop=True
        else:
            lex_dict.update(additions_to_lex)
            analyser.lexicon.update(additions_to_lex)

bootstrap()


df_test['scores'] = df_test['content'].apply(lambda review: analyser.polarity_scores(review))
df_test['compound']  = df_test['scores'].apply(lambda score_dict: score_dict['compound'])
df_test['Sentiment'] = df_test['compound'].apply(Sentimnt)

# var1 = df_test.groupby('Sentiment').count()['content'].reset_index().sort_values(by='content',ascending=False)
# print(var1.head())
# print(df[["content","class","Sentiment","scores","compound"]].head())

print(classification_report(df_test["class"], df_test["Sentiment"], target_names=class_names))
df_test["pred_class"] = df_test["Sentiment"]
filepath.parent.mkdir(parents=True, exist_ok=True)  #save evened distribution to a file
df_test[["content","class","pred_class"]].to_csv(filepath)  
