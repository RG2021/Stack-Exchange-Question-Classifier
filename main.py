# Importing Libraries.
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Reading the training file.
i=0
train=[]
topic=[]
with open('training.json','r') as f:
    for line in f:
        i=i+1
        if(i==1):
            continue
        j = json.loads(line)
        topic.append(j["topic"])
        train.append(j["excerpt"])

i=0
test=[]
test_output=[]

with open('test_input.json') as f:
    for line in f:
        i=i+1;
        if(i==1):
            continue
        j=json.loads(line)
        test.append(j['excerpt'])

with open('test_output.json') as f:
    for line in f:
        test_output.append(line)
        
train=np.array(train)
test=np.array(test)

vect = TfidfVectorizer(sublinear_tf=True, analyzer = 'word', ngram_range=(1,1), stop_words='english', max_features=None)
x_train = vect.fit_transform(train)
x_test = vect.transform(test)

le=preprocessing.LabelEncoder()
y_train = le.fit_transform(topic)

clf = MultinomialNB()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
y_predict=le.inverse_transform(y_predict)

test_output=np.array(test_output)
y_test=[]
for i in test_output:
    y_test.append(i.rstrip())
accuracy=accuracy_score(y_test, y_predict)
print(accuracy)
