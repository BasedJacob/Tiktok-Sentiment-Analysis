#runs classification report on all of the output files
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class_names = ['Negative', 'Neutral', 'Positive']

def printClassificationReports():
    if(Path('../output/lexicalBaseOut.csv').is_file()):
        lexicalBaseOut = pd.read_csv('../output/lexicalBaseOut.csv')
        print("lexicalBase:")
        print(classification_report(lexicalBaseOut["class"], lexicalBaseOut["pred_class"], target_names=class_names))
    else:
        print("no lexicalBaseOut.csv")
        
    if(Path('../output/bertBaseOut.csv').is_file()):
        bertBaseOut = pd.read_csv('../output/bertBaseOut.csv')
        print("bertBase:")
        print(classification_report(bertBaseOut["class"], bertBaseOut["pred_class"], target_names=class_names))
    else:
        print("no bertBaseOut.csv")
        
    if(Path('../output/lexicalBootstrapOut.csv').is_file()):
        lexicalBootstrapOut = pd.read_csv('../output/lexicalBootstrapOut.csv')
        print("lexicalBootstrap:")
        print(classification_report(lexicalBootstrapOut["class"], lexicalBootstrapOut["pred_class"], target_names=class_names))
    else:
        print("no lexicalBootstrap.csv")

    if(Path('../output/bertTrainingOut.csv').is_file()):
        bertTrainingOut = pd.read_csv('../output/bertTrainingOut.csv')
        print("bertTraining:")
        print(classification_report(bertTrainingOut["class"], bertTrainingOut["pred_class"], target_names=class_names))
    else:
        print("no bertTraining.csv")
    
printClassificationReports()
original_stdout = sys.stdout
with open('../output/classification_reports.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    printClassificationReports()
    sys.stdout = original_stdout 
    