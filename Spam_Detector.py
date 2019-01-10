import csv, os, math, operator, pickle, time, random, re

import numpy as np

import pandas as pd

data_dir = "C:\PythonData"

os.chdir(data_dir)

## Read data, select the first 2 columns, and rename these 2 columns as label, text
ColumnNames = ['label', 'text']
df = pd.read_table('./spam.csv', encoding='latin-1', sep=',', names = ColumnNames, usecols = [0,1],
                   skiprows = 1, header = None)

#Add another column and classify ham = 0 and spam = 1
df['label_num'] = df.label.map({'ham':0, 'spam':1})
print(df)

#Count: Ham 4825, spam 747
print(df.label.value_counts())
print(df.groupby('label').describe())

import string
import nltk
##nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer

##Clean and normalize text
def pre_process(text):
    text = text.lower()                                                         # lowercase all letters
    text = ''.join([t for t in text if t not in string.punctuation])            # remove all punctuations
    text = [t for t in text.split() if t not in stopwords.words('english')]     # remove stopwords
    st = Stemmer()                                                              # stem text
    text = [st.stem(t) for t in text]
    return text

##Test one of the messages
print(pre_process('The word \Checkmate\" in chess comes from the Persian phrase \"Shah Maat\" '
              'which means; \"the king is dead..\" Goodmorning.. Have a good day..:)"'))

##Convert each message to vectors, and assign weighted values(how frequent a term occurs in text) to these vectors.
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(analyzer = pre_process)
vector_output = vector.fit_transform(df['text'])

print (vector_output[0:10])
print(pd.DataFrame(vector_output.toarray()))


##Split data to training dataset and test dataset
from sklearn.model_selection import train_test_split
test_train, test_text, label_train, label_test = train_test_split(vector_output,df['label_num'],
                                                                  test_size=0.3, random_state=0)
##Fit in Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.2)
mnb.fit(test_train, label_train)
predictions = mnb.predict(test_text)

##Calculate number of test cases, and number of wrong predictions
count = 0
for i in range(len(label_test)):
    if label_test.iloc[i] != predictions[i]:
        count += 1
print('Number of test cases:', len(label_test))  #1672
print('Number of predictions:', count)           #35

## Calculate accuracy and F1-score
TP = 0      #Observation is positive, and is predicted to be positive
FN = 0      #Observation is positive, but is predicted negative
TN = 0      #Observation is negative, and is predicted to be negative
FP = 0      #Observation is negative, but is predicted positive

for i in range(len(label_test)):
    if label_test.iloc[i] == 1:
        if predictions[i] == 1:
            TP += 1
        else:
            FN +=1
    else:
        if predictions[i] == 1:
            FP += 1
        else:
            TN +=1
print("TP:", TP)  #212
print("FN:", FN)  #26
print("FP:", FP)  #9
print("TN:", TN)  #1425


Accuracy = (TP+TN)/(TP+FP+FN+TN)
print("Accuracy:", Accuracy)       #result: 0.979066985645933

Precision = TP/(TP+FP)
print("Precision:", Precision)

Recall = TP/(TP + FN)
print("Recall:", Recall)

F1Score = 2*(Recall * Precision) / (Recall + Precision)
print("F1Score:", F1Score)        #0.9237472766884532


##Use confusion matrix and classification report to verify the above calculation
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(label_test, predictions)
print("confusion matrix:")
print(confusion_matrix)

from sklearn.metrics import classification_report
print("classification report:")
print(classification_report(label_test,predictions))